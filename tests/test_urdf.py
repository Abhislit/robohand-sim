"""Verify the exported URDF is a faithful kinematic twin of the renderer.

These tests parse the generated URDF back out and run independent forward
kinematics on it, then compare link positions against `RobotHand.chain()`.
A drift between the OpenCV renderer and the URDF would mean the simulator and
the on-screen hand disagree, which is exactly the bug that would be invisible
until someone compared them - so it is asserted here.
"""

from __future__ import annotations

import math
import os
import sys
from xml.etree import ElementTree as ET

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robohand.math3d import euler_zyx, norm, rot_axis, vec
from robohand.robot_hand import FINGERS, FINGER_RIG, RobotHand
from robohand.urdf_export import JOINT_SUFFIX, build_urdf, joint_table

URDF_TOL = 1e-6


# --- an independent URDF FK implementation (deliberately not shared) --------
def _rpy_to_mat(rpy) -> np.ndarray:
    r, p, y = rpy
    return euler_zyx(y, p, r)


def _parse(urdf_text: str) -> dict:
    root = ET.fromstring(urdf_text)
    joints = {}
    for j in root.findall("joint"):
        name = j.get("name")
        origin = j.find("origin")
        axis = j.find("axis")
        limit = j.find("limit")
        joints[name] = {
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": [float(v) for v in origin.get("xyz", "0 0 0").split()],
            "rpy": [float(v) for v in origin.get("rpy", "0 0 0").split()],
            "axis": [float(v) for v in axis.get("xyz").split()] if axis is not None
                    else [0.0, 0.0, 1.0],
            "lower": float(limit.get("lower")) if limit is not None else 0.0,
            "upper": float(limit.get("upper")) if limit is not None else 0.0,
        }
    return joints


def urdf_fk(joints: dict, angles: dict[str, float]) -> dict[str, np.ndarray]:
    """Link positions in the root frame, using URDF's own conventions."""
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {"base_link": (vec(0, 0, 0), np.eye(3))}
    order = list(joints)
    for _ in range(len(order) + 1):
        for name in order:
            j = joints[name]
            if j["parent"] not in poses or j["child"] in poses:
                continue
            p_pos, p_rot = poses[j["parent"]]
            origin_rot = p_rot @ _rpy_to_mat(j["rpy"])
            origin_pos = p_pos + p_rot @ vec(*j["xyz"])
            angle = float(angles.get(name, 0.0))
            rot = origin_rot @ rot_axis(norm(vec(*j["axis"])), angle)
            poses[j["child"]] = (origin_pos, rot)
    return {k: v[0] for k, v in poses.items()}


# --- structure --------------------------------------------------------------
def test_urdf_has_15_revolute_joints_and_16_links():
    root = ET.fromstring(build_urdf())
    revolute = [j for j in root.findall("joint") if j.get("type") == "revolute"]
    assert len(revolute) == 15
    assert len(root.findall("link")) == 16
    names = {j.get("name") for j in revolute}
    for f in FINGERS:
        for s in JOINT_SUFFIX:
            assert f"{f}_{s}" in names


def test_urdf_is_valid_xml_with_a_robot_root():
    root = ET.fromstring(build_urdf())
    assert root.tag == "robot"
    assert root.get("name") == "robohand_right"


def test_single_root_link_and_a_connected_tree():
    joints = _parse(build_urdf())
    parents = [j["parent"] for j in joints.values()]
    links = {j["child"] for j in joints.values()} | {"base_link"}
    assert parents.count("base_link") == 5, "one chain per finger off the palm"
    assert len(links) == 16

    child_to_joint = {j["child"]: name for name, j in joints.items()}
    for link in links - {"base_link"}:
        seen, cur = set(), link
        while cur in child_to_joint:                # walk up to the parent
            assert cur not in seen, f"cycle at {cur}"
            seen.add(cur)
            cur = joints[child_to_joint[cur]]["parent"]
        assert cur == "base_link", f"{link} is not rooted at base_link"


def test_every_link_has_inertial_mass_and_geometry():
    root = ET.fromstring(build_urdf())
    for link in root.findall("link"):
        inertial = link.find("inertial")
        assert inertial is not None, f"{link.get('name')} has no <inertial>"
        assert float(inertial.find("mass").get("value")) > 0.0
        inert = inertial.find("inertia")
        assert float(inert.get("ixx")) > 0 and float(inert.get("iyy")) > 0
        assert float(inert.get("izz")) > 0
        assert link.find("visual") is not None, f"{link.get('name')} is invisible"


def test_joint_limits_match_the_rig():
    joints = _parse(build_urdf())
    for name, lo, hi, default in joint_table():
        j = joints[name]
        assert j["lower"] == pytest.approx(lo, abs=1e-9)
        assert j["upper"] == pytest.approx(hi, abs=1e-9)
        # Limits must be the rig's real anatomical maxima, not placeholders.
        assert 0.2 < hi < math.radians(130)


def test_joint_axes_are_unit_and_flexion_about_x():
    """Every joint flexes about local X. A left hand mirrors the geometry, so
    its axis is negated - this keeps positive joint values meaning "more curl"
    for both hands, which is what a controller wants."""
    for hand, expect in (("Right", [1.0, 0.0, 0.0]), ("Left", [-1.0, 0.0, 0.0])):
        joints = _parse(build_urdf(hand))
        for name, j in joints.items():
            assert j["type"] == "revolute"
            assert np.isclose(float(np.linalg.norm(j["axis"])), 1.0, atol=1e-9)
            assert j["axis"] == expect, f"{hand} {name} axis {j['axis']}"
            # Limits stay positive-going for both hands.
            assert j["lower"] == pytest.approx(0.0, abs=1e-9)


# --- the important part: does it move like the renderer? --------------------
@pytest.mark.parametrize("flex", [0.0, 0.25, 0.5, 0.85, 1.0])
def test_urdf_fk_matches_the_renderer_at_every_pose(flex):
    joints = _parse(build_urdf("Right"))
    angles = {f"{f}_{s}": flex * FINGER_RIG[f][2][i][2]
              for f in FINGERS for i, s in enumerate(JOINT_SUFFIX)}
    poses = urdf_fk(joints, angles)

    hand = RobotHand("Right")
    hand.set_pose({f"{f}_flex": flex for f in FINGERS})
    for finger in FINGERS:
        frames = hand.chain(finger, hand._signal)
        # frames[3] is the fingertip, past the last joint - only the 3 joints
        # have a corresponding URDF link.
        for i, suffix in enumerate(JOINT_SUFFIX):
            pos = hand.mirror_point(frames[i][0])
            expected = poses[f"{finger}_{suffix}_link"]
            assert np.allclose(pos, expected, atol=URDF_TOL), (
                f"{finger}/{suffix} at flex={flex}: "
                f"rig {np.round(pos, 6)} vs urdf {np.round(expected, 6)}"
            )


def test_urdf_fk_matches_the_renderer_for_the_left_hand():
    joints = _parse(build_urdf("Left"))
    angles = {f"{f}_{s}": 0.6 * FINGER_RIG[f][2][i][2]
              for f in FINGERS for i, s in enumerate(JOINT_SUFFIX)}
    poses = urdf_fk(joints, angles)
    hand = RobotHand("Left")
    hand.set_pose({f"{f}_flex": 0.6 for f in FINGERS})
    for finger in FINGERS:
        frames = hand.chain(finger, hand._signal)
        for i, suffix in enumerate(JOINT_SUFFIX):
            pos = hand.mirror_point(frames[i][0])
            assert np.allclose(pos, poses[f"{finger}_{suffix}_link"], atol=URDF_TOL)


def test_urdf_tip_travels_when_a_joint_rotates():
    """A dead joint would pass the pose tests above but never move anything."""
    joints = _parse(build_urdf())
    base = urdf_fk(joints, {})["index_dip_link"]
    bent = urdf_fk(joints, {"index_mcp": 0.5, "index_pip": 0.5,
                            "index_dip": 0.5})["index_dip_link"]
    assert np.linalg.norm(base - bent) > 0.01

    # An all-zero command must reproduce the neutral pose exactly.
    zero = urdf_fk(joints, {f"{f}_{s}": 0.0 for f in FINGERS for s in JOINT_SUFFIX})
    assert np.allclose(base, zero["index_dip_link"])


def test_curl_in_urdf_moves_every_finger():
    """Each finger must actually move, and move toward its own palm side.

    The four long fingers hang upward, so curling drops the tip; the thumb
    points down-and-out, so curling raises it. Both are correct - the test
    pins the direction per finger rather than assuming one sign.
    """
    joints = _parse(build_urdf())
    zero = urdf_fk(joints, {f"{f}_{s}": 0.0
                            for f in FINGERS for s in JOINT_SUFFIX})
    full = urdf_fk(joints, {f"{f}_{s}": FINGER_RIG[f][2][i][2]
                            for f in FINGERS for i, s in enumerate(JOINT_SUFFIX)})
    for finger in FINGERS:
        tip = f"{finger}_dip_link"
        moved = np.linalg.norm(full[tip] - zero[tip])
        assert moved > 0.02, f"{finger} barely moved ({moved:.4f} m) when curling"
        dy = full[tip][1] - zero[tip][1]
        assert dy < 0.0 if finger != "thumb" else dy > 0.0, (
            f"{finger} tip moved the wrong way in Y: {dy:+.4f} m"
        )


def test_export_is_deterministic():
    assert build_urdf("Right") == build_urdf("Right")
    assert build_urdf("Right") != build_urdf("Left")


def test_export_rejects_bad_handedness():
    with pytest.raises(ValueError):
        build_urdf("Middle")


def test_export_writes_a_parseable_file(tmp_path):
    path = tmp_path / "robohand.urdf"
    path.write_text('<?xml version="1.0"?>\n' + build_urdf())
    assert ET.parse(path).getroot().tag == "robot"


# --- ROS bridge payload -----------------------------------------------------
def test_ros_bridge_payload_has_all_five_fingers():
    """The bridge parses `finger=ratio` lines off stdin, so every finger must
    appear or it would silently hold a stale value."""
    from robohand.ros_bridge import RosHandBridge
    from robohand.robot_hand import FINGERS

    b = RosHandBridge(autostart=False)
    assert b.urdf.endswith("robohand_Right.urdf")
    assert b.hand == "Right"

    sent: list[str] = []
    b.proc = type("P", (), {"poll": lambda self: None, "stdin": _FakeStdin(sent)})()
    assert b.send({"index": 0.4, "middle": 1.0}) is True
    line = sent[0]
    for finger in FINGERS:
        assert f"{finger}=" in line
    # Missing fingers default to 0 rather than being omitted.
    assert "ring=0.0000" in line and "index=0.4000" in line


def test_ros_bridge_reports_a_dead_pipe():
    from robohand.ros_bridge import RosHandBridge

    b = RosHandBridge(autostart=False)
    b.proc = None
    assert b.send({"index": 0.5}) is False
    b.proc = type("P", (), {"poll": lambda self: 1, "stdin": None})()
    assert b.send({"index": 0.5}) is False


class _FakeStdin:
    def __init__(self, sink):
        self.sink = sink

    def write(self, text):
        self.sink.append(text)
        return len(text)

    def flush(self):
        pass
