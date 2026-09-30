"""Stream hand tracking into Gazebo (gz sim 10).

The Gazebo counterpart to :mod:`robohand.ros_bridge`. This starts
`tools/gazebo_pub.py`, which writes a target angle onto each joint's
`/model/<model>/joint/<joint>/cmd` topic; the world's `JointController` PID
then drives the real articulated body there. Gazebo is not playing a canned
animation - the fingers are physically pulled around, collide with one another
and are subject to gravity and inertia.

A subprocess is unavoidable here, exactly as for ROS: `gz.transport`'s
compiled extension is built for the system Python 3.14, while the app runs in a
3.12 virtualenv because MediaPipe has no 3.13+ wheels. So the publisher runs in
its own interpreter and the two halves talk over a stdin pipe, one line of
curls per camera frame.
"""

from __future__ import annotations

import os
import subprocess
import time

from .robot_hand import FINGERS

DEFAULT_MODEL = "robohand_Right"
DEFAULT_PUB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "gazebo_pub.py",
)
ROS_SETUP = "/opt/ros/lyrical/setup.bash"


class GazeboError(RuntimeError):
    pass


class GazeboHandBridge:
    """Owns the gz publisher subprocess and streams finger curls into it."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        hand: str = "Right",
        wrist: bool = True,
        publisher: str = DEFAULT_PUB,
        ros_setup: str = ROS_SETUP,
        root: str | None = None,
        autostart: bool = True,
    ) -> None:
        self.model = model
        self.hand = hand
        self.wrist = wrist
        self.publisher = publisher
        self.ros_setup = ros_setup
        self.root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.proc: subprocess.Popen | None = None
        self.log_path = os.path.join("/tmp", f"robohand_gazebo_{hand}.log")
        self._log = None
        self.sent = 0
        self.dropped = 0
        if autostart:
            self.start()

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        for path, hint in (
            (self.ros_setup, "ROS 2 setup"),
            (self.publisher, "gazebo_pub.py"),
        ):
            if not os.path.exists(path):
                raise GazeboError(f"{hint} not found at {path}")

        inner = (
            f"set -e; source {self.ros_setup} >/dev/null 2>&1; "
            f"export ROBOHAND_ROOT={self.root}; "
            f"exec python3 {self.publisher} --model {self.model} "
            f"--hand {self.hand} --wrist {1 if self.wrist else 0}"
        )
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            ["bash", "-c", inner],
            stdin=subprocess.PIPE,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        time.sleep(3.0)
        if self.proc.poll() is not None:
            raise GazeboError(
                f"gz publisher exited (rc={self.proc.returncode}). Log:\n"
                f"{self._tail()}"
            )
        tail = self._tail(3)
        if "0/1" in tail or "no joint controllers" in tail:
            print(f"[gazebo] {tail.strip()}")

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except OSError:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self) -> "GazeboHandBridge":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- streaming -----------------------------------------------------------
    def send(self, curls: dict[str, float], wrist: float = 0.0) -> bool:
        if self.proc is None or self.proc.poll() is not None:
            return False
        parts = [f"{f}={float(curls.get(f, 0.0)):.4f}" for f in FINGERS]
        parts.append(f"wrist={float(wrist):.4f}")
        try:
            self.proc.stdin.write(" ".join(parts) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            self.dropped += 1
            return False
        self.sent += 1
        return True

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _tail(self, lines: int = 8) -> str:
        if not os.path.exists(self.log_path):
            return "(no log)"
        with open(self.log_path) as fh:
            return "".join(fh.readlines()[-lines:])
