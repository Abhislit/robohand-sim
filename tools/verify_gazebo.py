#!/usr/bin/env python
"""Verify the webcam path really moves the Gazebo hand's physics.

Streams an open hand, a fist and a point through the gz-transport bridge and
watches the simulated fingertip positions in the world's dynamic_pose stream.
A correct result is a large, repeatable displacement per pose.
"""

import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robohand.gazebo_bridge import GazeboHandBridge  # noqa: E402
from robohand.robot_hand import FINGERS               # noqa: E402

TOPIC = "/world/robohand_world/dynamic_pose/info"
TIPS = ("index_dip_link", "middle_dip_link", "pinky_dip_link", "thumb_dip_link")


def link_positions():
    """World x/y/z of each link, from one dynamic_pose message."""
    out = subprocess.run(
        ["gz", "topic", "-e", "-t", TOPIC, "-n", "1"],
        capture_output=True, text=True, timeout=60).stdout
    # dynamic_pose is a repeated `pose { name: "x" ... position {x,y,z} }`.
    poses = re.findall(
        r'name:\s*"([^"]+)"\s+id:\s*\d+\s+position\s*\{([^}]*)\}', out)
    found = {}
    for name, blob in poses:
        nums = re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", blob)
        if len(nums) >= 3:
            found[name] = tuple(float(v) for v in nums[:3])
    return found


def dist(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def main():
    print("=== connect ===")
    try:
        bridge = GazeboHandBridge(model="robohand_Right", hand="Right")
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"  publisher pid {bridge.proc.pid}")
    print(f"  {bridge._tail(2).strip()}")

    ok = True
    try:
        poses = {
            "open": {f: 0.05 for f in FINGERS},
            "fist": {f: 1.0 for f in FINGERS},
            "point": {**{f: 1.0 for f in FINGERS}, "index": 0.05},
        }
        seen = {}
        for name, curls in poses.items():
            for _ in range(40):                    # hold the pose ~1s
                bridge.send(curls)
                time.sleep(0.02)
            time.sleep(2.5)                          # let the PID settle
            lp = link_positions()
            seen[name] = lp
            have = [t for t in TIPS if t in lp]
            print(f"=== {name}: {len(have)}/{len(TIPS)} tips found ===")
            for t in have:
                print(f"    {t:18} {tuple(round(v, 4) for v in lp[t])}")
            if len(have) < len(TIPS):
                ok = False

        if all(t in seen["open"] for t in TIPS):
            print("=== fingertip travel ===")
            for a, b in (("open", "fist"), ("fist", "point")):
                for t in TIPS:
                    d = dist(seen[a][t], seen[b][t])
                    print(f"    {a:>5} -> {b:<5} {t:18} {d:.4f} m")
            # The thumb curls across the palm, so its tip travels less than the
            # fingers' - it only has to move, not move as far.
            thumb = dist(seen["open"]["thumb_dip_link"], seen["fist"]["thumb_dip_link"])
            if thumb < 0.001:
                print("  KNOWN ISSUE: the thumb does not respond to Gazebo "
                      f"commands ({thumb*1000:.1f} mm). See README "
                      "'The thumb bug, and what it was'.")
            checks = [
                ("the four long fingers curl > 15 mm on a fist",
                 all(dist(seen["open"][t], seen["fist"][t]) > 0.015
                     for t in TIPS if t != "thumb_dip_link")),
                ("the thumb responds to commands at all", thumb > 0.001),
                ("point returns the index toward open",
                 dist(seen["fist"]["index_dip_link"],
                      seen["point"]["index_dip_link"])
                 > dist(seen["fist"]["index_dip_link"],
                        seen["open"]["index_dip_link"]) * 0.8),
                ("point keeps middle/pinky near the fist",
                 all(dist(seen["fist"][t], seen["point"][t]) < 0.05
                     for t in ("middle_dip_link", "pinky_dip_link"))),
            ]
            print("=== assertions ===")
            for label, passed in checks:
                print(f"  {'PASS' if passed else 'FAIL'}  {label}")
                ok = ok and passed

        # --- wrist sweep ------------------------------------------------
        # The wrist is checked separately from the finger poses because it
        # needs the arm anchored, and because a wrist that silently fails to
        # articulate looks identical to a wrist that is merely at its rest
        # angle. Measured on the fingertip, since that is what a user sees.
        print("=== wrist sweep ===")
        wrist_seen = {}
        for name, angle in (("flex", 1.0), ("neutral", 0.0), ("extend", -1.0)):
            for _ in range(120):                   # hold ~4s for the PID
                bridge.send({f: 0.05 for f in FINGERS}, wrist=angle)
                time.sleep(0.03)
            time.sleep(2.0)
            wrist_seen[name] = link_positions()
            print(f"  {name}: captured")

        tip = "index_dip_link"
        have_tip = all(tip in wrist_seen[n] for n in wrist_seen)
        if have_tip:
            flex = dist(wrist_seen["neutral"][tip], wrist_seen["flex"][tip])
            ext = dist(wrist_seen["neutral"][tip], wrist_seen["extend"][tip])
            print(f"  travel flex {flex*1000:.1f} mm, extend {ext*1000:.1f} mm")
            wrist_ok = flex > 0.01
        else:
            print(f"  FAIL  {tip} missing from the wrist sweep")
            wrist_ok = False
        for label, passed in [
            (f"wrist flexes the hand ({'' if not have_tip else f'{flex*1000:.0f} mm'})",
             wrist_ok),
        ]:
            print(f"  {'PASS' if passed else 'FAIL'}  {label}")
            ok = ok and passed
    finally:
        bridge.stop()
        print("=== publisher stopped ===")

    print(f"\nRESULT: {'PASS - Gazebo physics follows the hand' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
