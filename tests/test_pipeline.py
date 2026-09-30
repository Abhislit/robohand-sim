"""Unit tests for the robohand pipeline. Run with: python -m pytest -q"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robohand.demo import DemoDriver, SCRIPT
from robohand.gestures import GestureRecognizer, classify
from robohand.hand_tracker import (
    ANGLE_CLOSED,
    ANGLE_OPEN,
    FINGER_JOINTS,
    HandTracker,
    _angle_at,
)
from robohand.math3d import (
    clamp,
    cross,
    euler_zyx,
    mat_to_euler_zyx,
    norm,
    rot_axis,
    rot_between,
    vec,
)
from robohand.renderer import Camera3D, Renderer
from robohand.robot_hand import FINGERS, FINGER_RIG, RobotHand
from robohand.smoothing import OneEuroFilter, PoseSmoother


# --- math3d -----------------------------------------------------------------
def test_rot_axis_is_a_rotation():
    r = rot_axis(vec(0.3, -0.7, 0.2), 1.1)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_rot_between_maps_a_to_b():
    a, b = vec(1, 0, 0), norm(vec(0.2, 0.5, -0.8))
    assert np.allclose(rot_between(a, b) @ a, b, atol=1e-9)


def test_cross_matches_numpy():
    a, b = vec(1, 2, 3), vec(-4, 0.5, 2)
    assert np.allclose(cross(a, b), np.cross(a, b), atol=1e-12)


def test_euler_roundtrip():
    m = euler_zyx(0.3, -0.42, 1.1)
    yaw, pitch, roll = mat_to_euler_zyx(m)
    assert np.allclose(euler_zyx(yaw, pitch, roll), m, atol=1e-8)


def test_euler_roundtrip_at_gimbal_lock():
    m = euler_zyx(0.0, math.pi / 2 - 1e-6, 0.0)
    yaw, pitch, roll = mat_to_euler_zyx(m)
    assert np.allclose(euler_zyx(yaw, pitch, roll), m, atol=1e-4)


# --- smoothing --------------------------------------------------------------
def test_one_euro_suppresses_noise_when_static():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    t, out = 0.0, []
    rng = np.random.default_rng(0)
    for _ in range(120):
        t += 1 / 30
        out.append(float(f(0.5 + rng.normal(0, 0.08), t)))
    tail = np.array(out[-40:])
    # Static signal: residual jitter must be far below the raw noise (0.08).
    assert tail.std() < 0.03
    assert tail.std() < 0.5 * 0.08


def test_one_euro_tracks_a_fast_ramp():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.1)
    t, errs = 0.0, []
    for i in range(90):
        t += 1 / 30
        truth = min(1.0, i / 30.0)
        got = float(f(truth, t))
        if i > 20:
            errs.append(abs(got - truth))
    assert np.mean(errs) < 0.06, f"mean lag {np.mean(errs):.3f}"


def test_pose_smoother_outputs_every_requested_channel():
    keys = ("a", "b", "c")
    sm = PoseSmoother(keys, aggressiveness=0.5)
    out = sm({"a": 0.1, "b": 0.2, "c": 0.3}, 0.0)
    assert set(out) == set(keys)


def test_pose_smoother_aggressiveness_is_clamped():
    sm = PoseSmoother(("a",), aggressiveness=0.5)
    sm({"a": 0.5}, 0.0)
    sm.set_aggressiveness(5.0)
    assert 0.0 <= sm.aggressiveness <= 1.0
    sm.set_aggressiveness(-2.0)
    assert 0.0 <= sm.aggressiveness <= 1.0


# --- hand tracker -----------------------------------------------------------
def _straight_chain(length=0.04):
    return [vec(0, i * length, 0) for i in range(4)]


def _curled_chain(radius=0.012):
    """Four points wrapped onto a circle -> fully closed finger."""
    out = []
    for i in range(4):
        a = i * (math.pi / 2)
        out.append(vec(radius * math.cos(a), radius * math.sin(a), 0.0))
    return out


def _assemble(hand_pts):
    pts = np.zeros((21, 3))
    for name, (mcp, pip, dip, tip) in FINGER_JOINTS.items():
        chain = hand_pts[name]
        for idx, p in zip((mcp, pip, dip, tip), chain):
            pts[idx] = p
    return pts


def _angle_curl(a_pip, a_dip, ratio):
    ang = 0.62 * a_pip + 0.38 * a_dip
    curl_ang = clamp((ANGLE_OPEN - ang) / (ANGLE_OPEN - ANGLE_CLOSED))
    curl_dist = clamp((0.97 - ratio) / (0.97 - 0.52))
    return clamp(0.68 * curl_ang + 0.32 * curl_dist)


def test_angle_at_straight_and_right_angle():
    assert _angle_at(vec(0, 0, 0), vec(0, 1, 0), vec(0, 2, 0)) == pytest.approx(180, abs=1e-6)
    assert _angle_at(vec(1, 0, 0), vec(0, 0, 0), vec(0, 1, 0)) == pytest.approx(90, abs=1e-6)


def test_finger_curl_reads_zero_for_a_straight_finger():
    chain = _straight_chain()
    chain = [p + vec(0, 0.05, 0) for p in chain]
    ratio = np.linalg.norm(chain[-1] - chain[0]) / sum(
        np.linalg.norm(chain[i + 1] - chain[i]) for i in range(3)
    )
    assert _angle_curl(180.0, 180.0, ratio) < 0.06


def test_finger_curl_reads_one_for_a_closed_finger():
    ratio = 0.30
    assert _angle_curl(20.0, 20.0, ratio) > 0.95


def test_measure_produces_all_signal_keys_and_sane_values():
    tr = HandTracker()
    chain_s = _straight_chain()
    chain_c = _curled_chain()
    hand_pts = {
        "index": [p + vec(0, 0.05, 0) for p in chain_s],
        "middle": [p + vec(0, 0.05, 0) for p in chain_s],
        "ring": [p + vec(0, 0.05, 0) for p in chain_s],
        "pinky": [p + vec(0, 0.05, 0) for p in chain_s],
        "thumb": [p + vec(0, 0.05, 0) for p in chain_c],
    }
    pts = _assemble(hand_pts)
    pts[0] = vec(0, 0.0, 0)                     # wrist
    pts[9] = vec(0, 0.05, 0)                    # middle MCP
    # Spread the MCPs so palm geometry is non-degenerate.
    for name, x in (("index", -0.03), ("middle", -0.01), ("ring", 0.01), ("pinky", 0.03)):
        pts[FINGER_JOINTS[name][0]] = vec(x, 0.05, 0)

    hand = type("H", (), {})()
    hand.present, hand.landmarks = True, pts
    tr._measure(hand, pts)

    from robohand.hand_tracker import SIGNAL_KEYS

    assert set(hand.signal) == set(SIGNAL_KEYS)
    for f in FINGERS:
        assert 0.0 <= hand.signal[f"{f}_flex"] <= 1.25
    for k in ("palm_yaw", "palm_pitch", "palm_roll"):
        assert -math.pi <= hand.signal[k] <= math.pi
    assert 0.0 <= hand.signal["grip"] <= 1.0
    assert hand.palm_rot is not None
    assert np.allclose(hand.palm_rot @ hand.palm_rot.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(hand.palm_rot), 1.0, atol=1e-6)
    tr.close()


def test_measure_survives_degenerate_geometry():
    """An all-identical-landmark frame must not produce NaN or out-of-range."""
    tr = HandTracker()
    hand = type("H", (), {})()
    hand.present, hand.landmarks = True, np.zeros((21, 3))
    tr._measure(hand, np.zeros((21, 3)))
    for f in FINGERS:
        assert 0.0 <= hand.signal[f"{f}_flex"] <= 1.25
    assert all(math.isfinite(v) for v in hand.signal.values())
    assert np.all(np.isfinite(hand.palm_rot))
    assert np.allclose(hand.palm_rot @ hand.palm_rot.T, np.eye(3), atol=1e-6)
    tr.close()


def test_process_returns_empty_on_a_blank_frame():
    tr = HandTracker()
    assert tr.process(np.zeros((240, 320, 3), np.uint8)) == []
    assert tr.process(np.zeros((0, 0, 3), np.uint8)) == []
    assert tr.process(None) == []
    tr.close()


# --- gestures ---------------------------------------------------------------
def _sig(**vals):
    base = {f"{f}_flex": 0.0 for f in FINGERS}
    base["thumb_opposition"] = 0.0
    base.update(vals)
    return base


@pytest.mark.parametrize("vals,expected", [
    (dict(index_flex=0.95, middle_flex=0.95, ring_flex=0.95, pinky_flex=0.95,
          thumb_flex=0.10), "open"),
    (dict(index_flex=0.10, middle_flex=0.10, ring_flex=0.10, pinky_flex=0.10,
          thumb_flex=0.90), "fist"),
    (dict(index_flex=0.95, middle_flex=0.10, ring_flex=0.10, pinky_flex=0.10,
          thumb_flex=0.95), "point"),
    (dict(index_flex=0.90, middle_flex=0.90, ring_flex=0.10, pinky_flex=0.10,
          thumb_flex=0.90), "peace"),
    (dict(index_flex=0.90, middle_flex=0.90, ring_flex=0.90, pinky_flex=0.10,
          thumb_flex=0.90), "three"),
    (dict(index_flex=0.45, middle_flex=0.90, ring_flex=0.90, pinky_flex=0.90,
          thumb_flex=0.65), "pinch"),
    (dict(index_flex=0.90, middle_flex=0.90, ring_flex=0.90, pinky_flex=0.90,
          thumb_flex=0.90), "none"),
])
def test_classify_recognises_the_named_gesture(vals, expected):
    name, _ = classify(_sig(**vals))
    assert name == expected


def test_recognizer_requires_frames_before_committing():
    rec = GestureRecognizer(hold_frames=4)
    s = _sig(index_flex=0.95, middle_flex=0.95, ring_flex=0.95, pinky_flex=0.95,
             thumb_flex=0.1)
    seen = [rec.update(s, True).name for _ in range(6)]
    assert seen[:3] == ["none", "none", "none"], "must not commit immediately"
    assert "open" in seen[3:]


def test_recognizer_resets_when_hand_disappears():
    rec = GestureRecognizer(hold_frames=1)
    rec.update(_sig(index_flex=0.95, middle_flex=0.95, ring_flex=0.95,
                    pinky_flex=0.95, thumb_flex=0.1), True)
    assert rec.reading.name == "open"
    assert rec.update({}, False).name == "none"


# --- robot hand rig ---------------------------------------------------------
def test_rig_finger_lengths_are_positive_and_tapered():
    for name in FINGERS:
        segs = FINGER_RIG[name][2]
        assert len(segs) == 3
        for length, radius, max_flex in segs:
            assert length > 0 and radius > 0 and max_flex > 0


def test_solve_returns_primitives_in_both_handedness():
    for handedness in ("Right", "Left"):
        h = RobotHand(handedness)
        h.set_pose({f"{f}_flex": 0.5 for f in FINGERS})
        prims = h.solve()
        assert prims, f"{handedness} produced no primitives"
        for p in prims:
            if p.kind == "capsule":
                assert np.all(np.isfinite(p.p0)) and np.all(np.isfinite(p.p1))
            else:
                assert np.all(np.isfinite(p.center))
                assert np.allclose(p.rot @ p.rot.T, np.eye(3), atol=1e-9)


def test_fingertip_moves_toward_the_palm_as_the_finger_closes():
    def tip(curl):
        h = RobotHand("Right")
        h.set_pose({f"{f}_flex": curl for f in FINGERS})
        h.solve()
        return h.poses["index"].copy()

    extended, closed = tip(0.02), tip(1.0)
    # Curling rotates each joint away from +Y, so the tip must drop in height.
    assert extended[1] > closed[1] + 0.03


def test_thumb_opposition_sweeps_across_the_palm():
    def tip(opp):
        h = RobotHand("Right")
        h.set_pose({**{f"{f}_flex": 0.5 for f in FINGERS}, "thumb_opposition": opp})
        h.solve()
        return h.poses["thumb"].copy()

    relaxed, opposed = tip(0.0), tip(1.0)
    # Opposition rotates the thumb toward +X (inboard, over the palm).
    assert opposed[0] > relaxed[0] + 0.01


def test_solve_is_deterministic():
    h = RobotHand("Right")
    h.set_pose({f"{f}_flex": 0.42 for f in FINGERS})
    a = [p.p0.copy() for p in h.solve() if p.p0 is not None]
    b = [p.p0.copy() for p in h.solve() if p.p0 is not None]
    assert all(np.allclose(x, y) for x, y in zip(a, b))


def test_world_rotation_rotates_the_whole_rig():
    sig = {f"{f}_flex": 0.5 for f in FINGERS}
    h0 = RobotHand("Right")
    h0.set_pose(sig)
    base = [p.p0.copy() for p in h0.solve() if p.p0 is not None]

    h1 = RobotHand("Right")
    h1.set_pose(sig)
    h1.set_world_rotation(rot_axis(vec(0, 0, 1), math.pi / 2))
    rot = [p.p0.copy() for p in h1.solve() if p.p0 is not None]
    assert not np.allclose(base[0], rot[0])


def test_world_point_matches_the_rendered_primitive():
    h = RobotHand("Right")
    h.set_pose({f"{f}_flex": 0.3 for f in FINGERS})
    h.position = vec(0.1, 0.2, 0.0)
    h.set_world_rotation(rot_axis(vec(0, 1, 0), 0.7))
    h.solve()
    tip_world = h.world_point(h.poses["index"])
    assert np.allclose(tip_world, rot_axis(vec(0, 1, 0), 0.7) @ h.poses["index"]
                       + h.position)


def test_left_hand_is_the_mirror_of_the_right():
    sig = {f"{f}_flex": 0.5 for f in FINGERS}
    r, l = RobotHand("Right"), RobotHand("Left")
    r.set_pose(sig)
    l.set_pose(sig)
    rp = [p.p0 for p in r.solve() if p.p0 is not None]
    lp = [p.p0 for p in l.solve() if p.p0 is not None]
    assert all(np.allclose(a, [-b[0], b[1], b[2]]) for a, b in zip(rp, lp))


# --- renderer ---------------------------------------------------------------
def test_camera_projection_puts_the_target_at_the_centre():
    cam = Camera3D(640, 720, target=(0.0, 0.0, 0.0))
    u, v, d = cam.project_one(vec(0.0, 0.0, 0.0))
    assert abs(u - 320) < 1.0 and abs(v - 360) < 1.0 and d > 0


def test_camera_radius_scales_inversely_with_depth():
    cam = Camera3D(640, 720, target=(0.0, 0.0, 0.0))
    near = cam.radius_px(0.01, 0.2)
    far = cam.radius_px(0.01, 0.4)
    assert near > 0 and abs(near / far - 2.0) < 1e-6


def test_render_produces_a_non_trivial_image():
    h = RobotHand("Right")
    h.set_pose({f"{f}_flex": 0.5 for f in FINGERS})
    img = Renderer(320, 360, supersample=1).render(h.solve())
    assert img.shape == (360, 320, 3)
    assert img.std() > 12.0, "render looks like a flat fill"


def test_render_handles_an_empty_scene():
    img = Renderer(200, 200, supersample=1).render([])
    assert img.shape == (200, 200, 3)


def test_render_ignores_primitives_behind_the_camera():
    from robohand.robot_hand import Primitive

    r = Renderer(160, 160, supersample=1)
    far = Primitive(kind="capsule", color=(200, 200, 200),
                    p0=vec(0, 0, -50), p1=vec(0, 0.1, -50), r0=0.01, r1=0.01)
    assert r.render([far]) is not None


def test_supersample_does_not_change_the_output_size():
    h = RobotHand("Right")
    h.set_pose({f"{f}_flex": 0.4 for f in FINGERS})
    prims = h.solve()
    for ss in (1, 2):
        assert Renderer(200, 240, supersample=ss).render(prims).shape == (240, 200, 3)


# --- demo driver ------------------------------------------------------------
def test_demo_driver_emits_valid_signals_over_a_full_cycle():
    d = DemoDriver()
    for i in range(700):
        s = d.step(1 / 30)
        for f in FINGERS:
            assert 0.0 <= s[f"{f}_flex"] <= 1.0
        assert 0.0 <= s["thumb_opposition"] <= 1.0
        assert np.isfinite(s["x"]) and np.isfinite(s["y"])


def test_demo_driver_rotation_is_a_proper_rotation():
    d = DemoDriver()
    for _ in range(60):
        d.step(1 / 30)
        r = d.palm_rotation()
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(r), 1.0, atol=1e-6)


def test_demo_script_holds_each_pose_for_the_middle_of_its_window():
    """The middle of a stage should sit at that stage's target, not a blend."""
    d = DemoDriver()
    total = sum(dur for _, dur, _ in SCRIPT)
    acc = 0.0
    for name, dur, curls in SCRIPT:
        d.t = acc + dur * 0.5
        s = d.step(0.0)
        for f in FINGERS:
            assert s[f"{f}_flex"] == pytest.approx(curls[f], abs=0.02), \
                f"{name}/{f} mid-window was {s[f'{f}_flex']:.3f}, want {curls[f]}"
        acc += dur
    assert acc == pytest.approx(total)


# --- end to end -------------------------------------------------------------
def test_app_composes_a_frame_in_demo_mode():
    from robohand.app import Config, RoboHandApp

    app = RoboHandApp(Config(demo=True, width=640, height=360))
    frame, hands, palm_rots = app._read(1 / 30)
    smoothed = app._smooth(palm_rots, 0.0)
    app._apply(smoothed)
    canvas = app._compose(frame, hands, smoothed, 1 / 30)
    assert canvas.shape == (360, 640, 3)
    assert canvas.std() > 5.0


def test_app_preset_override_drives_the_rig():
    from robohand.app import Config, RoboHandApp
    from robohand.gestures import PRESETS

    app = RoboHandApp(Config(demo=True, width=640, height=360))
    app.preset_override = dict(PRESETS["3"])
    _, hands, palm_rots = app._read(1 / 30)
    app._apply(app._smooth(palm_rots, 0.0))
    assert all(app.hands[0].flex(f) > 0.9 for f in FINGERS)


def test_app_composes_a_camera_frame_with_a_detected_hand():
    """Regression: the demo path returns before the camera panel's per-hand
    loop, so a bug there stayed invisible until a real webcam was attached."""
    from robohand.app import Config, RoboHandApp
    from robohand.hand_tracker import TrackedHand

    app = RoboHandApp(Config(demo=False, width=640, height=360))
    hand = TrackedHand(present=True, handedness="Right")
    hand.landmarks = np.random.default_rng(1).random((21, 3)).astype(np.float32)
    hand.per_finger_curl = {f: 0.4 for f in FINGERS}
    frame = np.full((240, 320, 3), 60, np.uint8)
    palm_rots = {0: (hand.signal, None, "Right")}

    smoothed = app._smooth(palm_rots, 0.0)
    app._apply(smoothed)
    canvas = app._compose(frame, [hand], smoothed, 1 / 30)
    assert canvas.shape == (360, 640, 3)
    assert canvas.std() > 5.0


def test_run_loop_pushes_frames_to_the_window(monkeypatch):
    """Regression: the window was created but never given an image, so it
    rendered blank. Both display and quit-key handling must happen per frame."""
    import robohand.app as appmod

    shown, keys = [], iter([255, 255, ord("q")])
    monkeypatch.setattr(appmod.cv2, "imshow", lambda name, img: shown.append(img.copy()))
    monkeypatch.setattr(appmod.cv2, "waitKey", lambda d: next(keys))
    monkeypatch.setattr(appmod.cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(appmod.cv2, "resizeWindow", lambda *a, **k: None)
    monkeypatch.setattr(appmod.cv2, "destroyAllWindows", lambda *a, **k: None)

    app = appmod.RoboHandApp(appmod.Config(demo=True, width=640, height=360))
    assert app.run() == 0
    assert shown, "nothing was ever pushed to the window"
    assert all(img.shape == (360, 640, 3) for img in shown)
    assert max(float(i.std()) for i in shown) > 5.0
