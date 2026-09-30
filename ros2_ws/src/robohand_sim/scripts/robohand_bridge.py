#!/usr/bin/env python3
"""Publish the robohand's 15 joint angles on /joint_states.

Bridges the webcam app to ROS 2. Two modes:

  --source demo          drive a scripted open/close cycle (no camera)
  --source /joint_states replay a joint vector read from stdin

The signal values are the same 0..1 flexion ratios the OpenCV renderer uses;
they are scaled by each joint's real maximum, which is read straight out of the
URDF, so the published angles are the physical ones the hand would need.
"""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SUFFIX = ("mcp", "pip", "dip")


def load_joint_limits(urdf_path):
    root = ET.parse(urdf_path).getroot()
    limits = {}
    for j in root.findall("joint"):
        if j.get("type") != "revolute":
            continue
        lim = j.find("limit")
        limits[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return limits


def demo_signal(t):
    """A smooth open -> fist -> open cycle, matching the app's demo script."""
    phase = (t % 14.0) / 14.0
    cycle = 0.5 - 0.5 * math.cos(phase * 2.0 * math.pi)
    waves = {
        "thumb": 0.15 + 0.85 * cycle,
        "index": cycle,
        "middle": max(0.0, cycle - 0.05),
        "ring": max(0.0, cycle - 0.10),
        "pinky": max(0.0, cycle - 0.15),
    }
    # Each finger's curl is shared across its three joints, weighted the same
    # way the rig does, so the pose matches the OpenCV renderer.
    return waves


class RoboHandBridge(Node):
    def __init__(self, urdf_path, hand, rate_hz, source):
        super().__init__("robohand_bridge")
        self.limits = load_joint_limits(urdf_path)
        missing = [
            f"{f}_{s}" for f in FINGERS for s in SUFFIX if f"{f}_{s}" not in self.limits
        ]
        if missing:
            self.get_logger().error(f"URDF is missing joints: {missing}")
            raise SystemExit(1)

        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.t = 0.0
        self.dt = 1.0 / rate_hz
        self.source = source
        self._last = {f: 0.0 for f in FINGERS}
        self._open_stdin()
        self.get_logger().info(
            f"robohand_bridge publishing {len(self.limits)} joints "
            f"on /joint_states ({hand} hand, source={source})"
        )

    def _open_stdin(self):
        """Non-blocking stdin.

        A webcam may only produce ~10 Hz while this timer runs at 30 Hz, and a
        plain readline() would stall the whole executor - freezing RViz between
        camera frames. So drain whatever has arrived, keep the newest complete
        line, and hold the previous value on the ticks where nothing came.
        """
        self._buf = ""
        try:
            fd = sys.stdin.fileno()
        except (AttributeError, ValueError):
            fd = None
        self._fd = fd
        if fd is not None:
            os.set_blocking(fd, False)

    def _poll_stdin(self):
        if self._fd is None:
            return self._last
        try:
            chunk = os.read(self._fd, 4096)
        except (BlockingIOError, InterruptedError):
            chunk = b""
        except OSError:
            chunk = b""
        if chunk:
            self._buf += chunk.decode("ascii", "ignore")
        # Keep only the trailing incomplete line.
        *complete, self._buf = self._buf.split("\n")
        for line in complete:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = {k: float(v) for k, v in
                          (tok.split("=") for tok in line.split()) if v}
            except ValueError:
                continue
            for finger in FINGERS:
                if finger in parsed:
                    self._last[finger] = min(1.25, max(0.0, parsed[finger]))
        return self._last

    def _tick(self):
        self.t += self.dt
        if self.source == "demo":
            waves = demo_signal(self.t)
        else:
            waves = self._poll_stdin()

        names, positions = [], []
        for finger in FINGERS:
            curl = float(waves.get(finger, 0.0))
            for suffix in SUFFIX:
                name = f"{finger}_{suffix}"
                lo, hi = self.limits[name]
                names.append(name)
                positions.append(lo + (hi - lo) * curl)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = positions
        self.pub.publish(msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--hand", default="Right", choices=("Right", "Left"))
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--source", default="demo", choices=("demo", "stdin"))
    # `ros2 launch` appends `--ros-args -r __node:=...`, which argparse would
    # reject outright. parse_known_args lets rclpy handle its own remaps.
    args, _unknown = ap.parse_known_args()

    rclpy.init()
    node = RoboHandBridge(args.urdf, args.hand, args.rate, args.source)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
