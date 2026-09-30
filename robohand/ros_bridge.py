"""Send live hand tracking to a ROS 2 simulator.

The webcam app runs in its own Python 3.12 virtualenv (MediaPipe has no 3.13+
wheels) while ROS 2 here runs on the system Python 3.14, so the two cannot share
an interpreter. Instead this module spawns `robohand_bridge.py` inside a shell
that sources the ROS environment, and feeds it one line of flexion values per
frame over stdin:

    index=0.12 middle=0.55 ring=0.61 pinky=0.60 thumb=0.08

The bridge scales each 0-1 ratio by that joint's real maximum, read out of the
URDF, and publishes `/joint_states`. RViz, Gazebo and MoveIt then show the hand.
"""

from __future__ import annotations

import os
import subprocess
import time

from .robot_hand import FINGERS

DEFAULT_ROS_SETUP = "/opt/ros/lyrical/setup.bash"


def default_workspace() -> str:
    """The bundled ROS 2 workspace, next to the robohand package."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "ros2_ws")


def default_urdf(ws: str | None = None, hand: str = "Right") -> str:
    ws = ws or default_workspace()
    return os.path.join(ws, "src", "robohand_sim", "urdf", f"robohand_{hand}.urdf")


def find_bridge_script(ws: str | None = None) -> str:
    """The installed bridge, falling back to the source tree before a build."""
    ws = ws or default_workspace()
    installed = os.path.join(ws, "install", "robohand_sim", "lib", "robohand_sim",
                             "robohand_bridge.py")
    if os.path.exists(installed):
        return installed
    src = os.path.join(ws, "src", "robohand_sim", "scripts", "robohand_bridge.py")
    if os.path.exists(src):
        return src
    raise FileNotFoundError(
        f"robohand_bridge.py not found under {ws}. Run `colcon build` in {ws}."
    )


class RosBridgeError(RuntimeError):
    pass


class RosHandBridge:
    """Owns the bridge subprocess and streams finger curls into it."""

    def __init__(
        self,
        urdf: str | None = None,
        hand: str = "Right",
        workspace: str | None = None,
        ros_setup: str = DEFAULT_ROS_SETUP,
        rate: float = 30.0,
        autostart: bool = True,
    ) -> None:
        self.workspace = workspace or default_workspace()
        self.hand = hand
        self.urdf = urdf or default_urdf(self.workspace, hand)
        self.ros_setup = ros_setup
        self.rate = float(rate)
        self.proc: subprocess.Popen | None = None
        self.log_path: str | None = None
        self._log = None
        self._sent = 0
        self._dropped = 0
        if autostart:
            self.start()

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        if not os.path.exists(self.ros_setup):
            raise RosBridgeError(f"ROS 2 setup not found at {self.ros_setup}")
        if not os.path.exists(self.urdf):
            raise RosBridgeError(
                f"URDF not found at {self.urdf}. Generate it with:\n"
                f"  python -m robohand --export-urdf {self.urdf} --hand {self.hand}"
            )
        script = find_bridge_script(self.workspace)
        install = os.path.join(self.workspace, "install", "setup.bash")

        # A shell wrapper is what makes the two interpreters work together: it
        # sources the ROS environment, then execs the system python on the
        # bridge script, so rclpy and MediaPipe never have to coexist.
        inner = (
            f"set -e; source {self.ros_setup} >/dev/null 2>&1; "
            + (f"source {install} >/dev/null 2>&1; " if os.path.exists(install) else "")
            + f"exec python3 {script} --urdf {self.urdf} "
              f"--hand {self.hand} --source stdin --rate {self.rate}"
        )

        self.log_path = os.path.join(
            "/tmp", f"robohand_bridge_{self.hand}.log"
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
        # Give ROS a moment to initialise before the first frame is pushed,
        # otherwise the bridge can die on a half-written first line.
        time.sleep(2.5)
        if self.proc.poll() is not None:
            tail = self._tail_log()
            raise RosBridgeError(
                f"bridge exited immediately (rc={self.proc.returncode}). Log:\n{tail}"
            )

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

    def __enter__(self) -> "RosHandBridge":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- streaming -----------------------------------------------------------
    def send(self, curls: dict[str, float]) -> bool:
        """Push one frame of flexion values. Returns False if the pipe is gone."""
        if self.proc is None or self.proc.poll() is not None:
            return False
        payload = " ".join(
            f"{f}={float(curls.get(f, 0.0)):.4f}" for f in FINGERS
        )
        try:
            self.proc.stdin.write(payload + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            self._dropped += 1
            return False
        self._sent += 1
        return True

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def stats(self) -> dict:
        return {"frames_sent": self._sent, "frames_dropped": self._dropped}

    def _tail_log(self, lines: int = 12) -> str:
        if not self.log_path or not os.path.exists(self.log_path):
            return "(no log)"
        with open(self.log_path) as fh:
            return "".join(fh.readlines()[-lines:])
