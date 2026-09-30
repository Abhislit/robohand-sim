#!/usr/bin/env python3
"""Publish hand curls to Gazebo joint controllers. Runs on the ROS 2 python.

Reads one line per frame on stdin:

    thumb=0.08 index=0.41 middle=0.72 ring=0.57 pinky=0.61 wrist=0.0

and writes the corresponding target angle (radians) onto each joint's
`/model/<model>/joint/<joint>/cmd` gz-transport topic, where the world's
JointController PID drives the real body to it.

This lives outside the robohand venv on purpose: `gz.transport`'s compiled
extension is built for the system Python 3.14, while the app runs in a 3.12
virtualenv because MediaPipe has no 3.13+ wheels. Same split as the ROS bridge.
"""

import argparse
import os
import select
import sys
import time

# The venv-side module that holds the joint limits, so the two halves agree.
sys.path.insert(0, os.environ.get("ROBOHAND_ROOT", "") or os.getcwd())

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SUFFIXES = ("mcp", "pip", "dip")
WRIST = "wrist_pitch"


def load_double():
    from gz.msgs import double_pb2
    return double_pb2.Double


# gz.transport's publish_raw takes (serialised_message, message_type_name) as
# two str arguments - not a protobuf object - so the type name has to travel
# with the payload and the protobuf bytes are mapped 1:1 through latin-1
# (the only encoding that survives every byte value).
DOUBLE_TYPE = "gz.msgs.Double"


def make_encoder():
    """Serialise a Double into the (str, type name) pair publish_raw wants."""
    Double = load_double()

    def _enc(value: float):
        msg = Double()
        msg.data = float(value)
        return msg.SerializeToString().decode("latin-1"), DOUBLE_TYPE

    return _enc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="robohand_Right")
    ap.add_argument("--hand", default="Right", choices=("Right", "Left"))
    ap.add_argument("--rate", type=float, default=60.0)
    ap.add_argument("--wrist", type=int, default=1)
    args = ap.parse_args()

    from gz.transport import Node
    from robohand.gazebo_world import command_topics, joint_limits

    limits = joint_limits(args.hand)
    Double = load_double()
    _enc = make_encoder()
    node = Node()

    topics = command_topics(args.model, args.hand)
    pubs = {}
    for name, topic in topics.items():
        if name == WRIST and not args.wrist:
            continue
        pubs[name] = node.advertise(topic, Double)
        time.sleep(0.004)

    deadline = time.time() + 5.0
    while time.time() < deadline:
        if any(p.has_connections() for p in pubs.values()):
            break
        time.sleep(0.05)
    live = sum(1 for p in pubs.values() if p.has_connections())
    print(f"[gazebo] streaming {live}/{len(pubs)} joints to model "
          f"'{args.model}' ({args.hand} hand)", flush=True)
    if live == 0:
        print("[gazebo] no joint controllers are listening - is the world "
              "running with the arm attached?", flush=True)

    last = {f: 0.0 for f in FINGERS}
    wrist = 0.0
    buf = ""
    while True:
        # Non-blocking drain: hold the last pose on frames where the camera
        # (at ~10 Hz) has nothing new to say.
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 1.0 / args.rate)
        except (OSError, ValueError):
            break
        if ready:
            chunk = os.read(sys.stdin.fileno(), 4096)
            if not chunk:
                break                      # producer closed the pipe
            buf += chunk.decode("ascii", "ignore")
        *complete, buf = buf.split("\n")
        for line in complete:
            for tok in line.split():
                if "=" not in tok:
                    continue
                k, v = tok.split("=", 1)
                try:
                    val = max(-2.0, min(2.0, float(v)))
                except ValueError:
                    continue
                if k == "wrist":
                    wrist = val
                elif k in last:
                    last[k] = val

        for finger in FINGERS:
            ratio = max(0.0, min(1.25, last[finger]))
            for suffix in SUFFIXES:
                name = f"{finger}_{suffix}"
                lo, hi = limits[name]
                pubs[name].publish_raw(*_enc(lo + (hi - lo) * ratio))
        if args.wrist:
            pubs[WRIST].publish_raw(*_enc(max(-1.15, min(1.15, wrist * 1.15))))


if __name__ == "__main__":
    main()
