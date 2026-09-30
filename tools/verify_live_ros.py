#!/usr/bin/env python
"""End-to-end check: does RosHandBridge really move the ROS 2 model?

Starts the bridge, streams a scripted curl sequence, and asserts that
/joint_states reflects the curls that were sent. Run with the simulation
already up (tools/launch_ros2_sim.sh) or on its own.
"""

import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robohand.robot_hand import FINGERS          # noqa: E402
from robohand.ros_bridge import RosBridgeError, RosHandBridge  # noqa: E402


def read_joint_positions():
    out = subprocess.run(
        ["ros2", "topic", "echo", "/joint_states", "--once", "--field", "position"],
        capture_output=True, text=True, timeout=30).stdout
    return [float(m) for m in re.findall(r"-?\d+\.\d+(?:e[-+]?\d+)?", out)]


def main():
    print("=== 1. start the bridge ===")
    try:
        bridge = RosHandBridge(hand="Right")
    except (RosBridgeError, FileNotFoundError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"  spawned, pid {bridge.proc.pid}, log {bridge.log_path}")

    ok = True
    try:
        # 2. open hand
        print("=== 2. stream an OPEN hand (all curls 0.05) ===")
        for _ in range(20):
            bridge.send({f: 0.05 for f in FINGERS})
            time.sleep(0.05)
        time.sleep(2.0)
        open_pos = read_joint_positions()
        print(f"  got {len(open_pos)} values: {[round(v, 3) for v in open_pos[:6]]}")

        # 3. closed fist
        print("=== 3. stream a FIST (all curls 1.0) ===")
        for _ in range(20):
            bridge.send({f: 1.0 for f in FINGERS})
            time.sleep(0.05)
        time.sleep(2.0)
        fist_pos = read_joint_positions()
        print(f"  got {len(fist_pos)} values: {[round(v, 3) for v in fist_pos[:6]]}")

        # 4. only the index extended
        print("=== 4. stream a POINT (index 0.05, rest 1.0) ===")
        curls = {f: 1.0 for f in FINGERS}
        curls["index"] = 0.05
        for _ in range(20):
            bridge.send(curls)
            time.sleep(0.05)
        time.sleep(2.0)
        point_pos = read_joint_positions()
        print(f"  got {len(point_pos)} values: {[round(v, 3) for v in point_pos[:6]]}")

        # Joint order is thumb_mcp,pip,dip, index_mcp,pip,dip, middle_...,
        # ring_..., pinky_...  Joints of the same kind must be compared against
        # each other: their maxima differ (mcp 1.658, pip 1.833, dip 1.134), so
        # an absolute cross-joint comparison would be meaningless.
        MCP = {"thumb": 0, "index": 3, "middle": 6, "ring": 9, "pinky": 12}

        checks = [
            ("15 values each read",
             len(open_pos) == 15 and len(fist_pos) == 15 and len(point_pos) == 15),
            ("fist > open (curl actually applied)",
             all(f > o for f, o in zip(fist_pos, open_pos))),
            ("fist hits each joint's URDF maximum",
             all(abs(f - m) < 0.02 for f, m in zip(
                 fist_pos, [0.7330, 0.8029, 1.0821, 1.6581, 1.8326, 1.1345,
                            1.6581, 1.8326, 1.1345, 1.6581, 1.8326, 1.1345,
                            1.6581, 1.7453, 1.1345]))),
            ("point: index straight, middle curled",
             point_pos[MCP["index"]] < 0.25 and point_pos[MCP["middle"]] > 1.0),
            ("point: index_mcp is the straightest of the four MCPs",
             point_pos[MCP["index"]] == min(
                 point_pos[MCP[f]] for f in ("index", "middle", "ring", "pinky"))),
            ("point: middle/ring/pinky MCPs all match the fist",
             all(abs(point_pos[MCP[f]] - fist_pos[MCP[f]]) < 0.02
                 for f in ("middle", "ring", "pinky"))),
            ("nothing exceeded the URDF limit",
             all(0.0 <= v <= 1.9 for v in fist_pos + point_pos + open_pos)),
        ]
        print("=== 5. assertions ===")
        for name, passed in checks:
            print(f"  {'PASS' if passed else 'FAIL'}  {name}")
            ok = ok and passed
        print(f"  frames sent: {bridge.stats['frames_sent']}, "
              f"dropped: {bridge.stats['frames_dropped']}")
    finally:
        bridge.stop()
        print("=== bridge stopped ===")

    print("\nRESULT:", "PASS - webcam curls drive the ROS 2 model" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
