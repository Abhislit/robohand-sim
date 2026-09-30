"""Main application: webcam hand tracking driving a 3D robot hand simulation.

    left panel   camera feed with the tracked skeleton + per-finger curl bars
    right panel  the simulated robot hand, rendered in 3D
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .demo import DemoDriver
from .gestures import PRESET_NAMES, PRESETS, GestureRecognizer
from .hand_tracker import (
    SIGNAL_KEYS,
    HandTracker,
    TrackedHand,
    draw_hand_overlay,
)
from .math3d import clamp
from .renderer import Renderer
from .robot_hand import FINGERS, RobotHand
from .smoothing import FpsMeter, PoseSmoother

FINGER_LABEL = {"thumb": "THM", "index": "IDX", "middle": "MID",
                "ring": "RNG", "pinky": "PNK"}

WHITE = (238, 238, 240)
DIM = (140, 144, 150)
CYAN = (214, 176, 74)
GREEN = (122, 214, 110)
ORANGE = (86, 140, 240)
YELLOW = (96, 214, 240)


@dataclass
class Config:
    source: int | str = 0
    width: int = 1280
    height: int = 720
    camera_width: int = 960
    camera_height: int = 720
    demo: bool = False
    mirror: bool = True
    no_brightness: bool = False
    gain: float = 1.35
    smoothing: float = 0.45
    move_gain: float = 0.30
    rotate: float = 1.0
    max_hands: int = 2
    model_complexity: int = 1
    record: str = ""
    out_dir: str = "captures"
    window: str = "robohand"
    fov: float = 38.0
    dark_threshold: float = 26.0
    ros_urdf: str = ""            # non-empty => stream this hand to ROS 2
    ros_hand: str = "Right"
    ros_workspace: str = ""


class RoboHandApp:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.running = True
        self.paused = False
        self.show_grid = True
        self.follow = True
        self.preset_override: dict | None = None
        self.screenshot_index = 0

        left_w = cfg.width // 2
        right_w = cfg.width - left_w
        self.left_w, self.right_w = left_w, right_w

        # Two hands means ~2x the primitives; drop the antialiasing supersample
        # to keep the frame rate usable.
        self.renderer = Renderer(right_w, cfg.height, supersample=2 if cfg.max_hands == 1 else 1)
        self.tracker: HandTracker | None = None
        self.demo = DemoDriver() if cfg.demo else None
        self.gesture = GestureRecognizer()
        self.fps = FpsMeter()
        self.hands: list[RobotHand] = []
        self.smoothers: dict[int, PoseSmoother] = {}
        self.last_t = time.perf_counter()
        self.cam_target = np.array([0.0, -0.012, 0.0])
        self.writer = None
        self._cam = None
        self.notice = ""
        self.notice_until = 0.0
        self.too_dark = False
        self.bridge = None
        # Without this, SIGTERM kills the process mid-frame and any --record
        # file is left without its index and unplayable.
        signal.signal(signal.SIGTERM, self._signal_stop)

    # -- ROS 2 streaming -----------------------------------------------------
    def _start_ros_bridge(self) -> None:
        from .ros_bridge import RosBridgeError, RosHandBridge

        try:
            self.bridge = RosHandBridge(
                urdf=self.cfg.ros_urdf,
                hand=self.cfg.ros_hand,
                workspace=self.cfg.ros_workspace or None,
            )
        except (RosBridgeError, FileNotFoundError) as exc:
            self.notice = f"ROS bridge failed: {exc}"
            self.notice_until = time.time() + 12.0
            print(f"\n[ros] {exc}\n[ros] the app will keep running without ROS output")
            self.bridge = None
            return
        print(f"[ros] streaming {self.cfg.ros_hand} hand -> /joint_states")
        print(f"[ros] bridge log: {self.bridge.log_path}")

    def _publish_ros(self, smoothed) -> None:
        """Forward the primary hand's flexion ratios to the ROS 2 bridge."""
        if self.bridge is None or not smoothed:
            return
        signal = smoothed[sorted(smoothed)[0]][0]
        curls = {f: float(signal.get(f"{f}_flex", 0.0)) for f in FINGERS}
        if not self.bridge.send(curls):
            if self.bridge.proc is not None:
                self.notice = "ROS bridge died - simulation stopped"
                self.notice_until = time.time() + 8.0
                self.bridge = None

    # -- setup ---------------------------------------------------------------
    def _signal_stop(self, *_args) -> None:
        self.running = False

    def _open_source(self):
        if self.demo is not None:
            return None
        src = self.cfg.source
        cap = cv2.VideoCapture(src if isinstance(src, str) else int(src))
        if not cap.isOpened():
            raise SystemExit(
                f"cannot open camera source {src!r}. "
                "Try `python -m robohand --list-cameras`."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
        return cap

    def _setup_record(self) -> None:
        if not self.cfg.record:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.cfg.record)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.cfg.record, fourcc, 30.0, (self.cfg.width, self.cfg.height)
        )
        if not self.writer.isOpened():
            print(f"[warn] could not open {self.cfg.record} for writing")
            self.writer = None

    # -- main loop -----------------------------------------------------------
    def run(self) -> int:
        cfg = self.cfg
        self._cam = self._open_source()
        self._setup_record()
        if self.demo is None:
            self.tracker = HandTracker(
                max_hands=cfg.max_hands, model_complexity=cfg.model_complexity,
                gain=cfg.gain, mirror=cfg.mirror,
            )
            print("show your hand to the camera - press H in the window for help")
        if self.cfg.ros_urdf:
            self._start_ros_bridge()

        last = time.perf_counter()
        try:
            while self.running:
                now = time.perf_counter()
                dt = clamp(now - last, 1e-3, 0.1)
                last = now
                t = time.time()

                frame, hands, palm_rots = self._read(dt)
                smoothed = self._smooth(palm_rots, t)
                self._apply(smoothed)

                self._publish_ros(smoothed)
                canvas = self._compose(frame, hands, smoothed, dt)
                cv2.imshow(self.cfg.window, canvas)
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key, canvas):
                    break
                if self.writer is not None:
                    self.writer.write(canvas)
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            self._teardown()
        return 0

    def _teardown(self) -> None:
        if self.writer is not None:
            self.writer.release()
        if self.tracker is not None:
            self.tracker.close()
        if self._cam is not None:
            self._cam.release()
        if self.bridge is not None:
            self.bridge.stop()
        cv2.destroyAllWindows()
        print("bye")

    # -- per-frame stages ----------------------------------------------------
    def _read(self, dt: float):
        """Returns (display frame, tracked hands, per-hand palm rotations)."""
        if self.demo is not None:
            signal = self.demo.step(dt)
            fake = TrackedHand(present=True, handedness="Right")
            fake.signal = signal
            fake.per_finger_curl = {
                f: float(signal.get(f"{f}_flex", 0.0)) for f in FINGERS
            }
            return None, [fake], {0: (signal, self.demo.palm_rotation(), "Right")}

        ok, frame = self._cam.read()
        if not ok or frame is None:
            self.notice = "camera returned no frame - stopping"
            self.running = False
            blank = np.zeros(
                (self.cfg.camera_height, self.cfg.camera_width, 3), np.uint8
            )
            return blank, [], {}

        if not self.cfg.no_brightness:
            frame = _auto_brightness(frame)

        self.too_dark = bool(self.demo is None and float(frame.mean()) < self.cfg.dark_threshold)
        hands = self.tracker.process(frame)
        palm_rots = {i: (h.signal, h.palm_rot, h.handedness)
                     for i, h in enumerate(hands)}
        return frame, hands, palm_rots

    def _smooth(self, palm_rots, t: float):
        """One Euro-filter every joint signal so the rig moves like a servo."""
        result = {}
        for i, (sig, palm_rot, handed) in palm_rots.items():
            key = id(palm_rot)
            sm = self.smoothers.get(key)
            if sm is None:
                sm = PoseSmoother(SIGNAL_KEYS, aggressiveness=self.cfg.smoothing)
                sm(sig, t, seen=True)            # prime to avoid a first-frame jump
                self.smoothers[key] = sm
            result[i] = (sm(sig, t, seen=True), palm_rot, handed)
        self.gesture.update(
            next(iter(palm_rots.values()))[0] if palm_rots else {}, present=bool(palm_rots)
        )
        return result

    def _apply(self, smoothed) -> None:
        """Drive the RobotHand rigs from the smoothed signals."""
        wanted: list[RobotHand] = []
        for n, (i, (sig, palm_rot, handed)) in enumerate(sorted(smoothed.items())):
            hand = RobotHand(handedness=handed)
            pose = dict(sig)

            if self.preset_override is not None:
                pose.update({f"{f}_flex": v for f, v in self.preset_override.items()})
                pose["thumb_opposition"] = float(
                    np.clip(0.2 + 0.8 * self.preset_override.get("index", 0.0), 0.0, 1.0)
                )

            hand.set_pose(pose)
            g = self.cfg.move_gain
            off = np.array([
                (pose.get("x", 0.5) - 0.5) * g,
                -(pose.get("y", 0.5) - 0.5) * g * 1.15,
                0.0,
            ])
            # Two hands sit side by side instead of overlapping.
            if n > 0:
                off[0] += 0.13 * n
            hand.position = off

            rot = None
            if palm_rot is not None and self.cfg.rotate > 0.0:
                rot = _blend_rot(np.eye(3), np.asarray(palm_rot, float),
                                 self.cfg.rotate)
            hand.set_world_rotation(rot)
            hand.spread = float(clamp(sig.get("openness", 0.0) * 0.6, 0.0, 1.0))
            wanted.append(hand)
        self.hands = wanted

    def _compose(self, frame, hands, smoothed, dt: float) -> np.ndarray:
        canvas = np.zeros((self.cfg.height, self.cfg.width, 3), np.uint8)
        self._draw_camera(canvas, frame, hands, dt)
        self._draw_robot(canvas, dt)
        self._draw_hud(canvas, hands, smoothed, dt)
        return canvas

    # -- panels --------------------------------------------------------------
    def _draw_camera(self, canvas, frame, hands, dt: float) -> None:
        panel = canvas[:, : self.left_w]
        if frame is None:
            _text(panel, (self.left_w // 2, self.cfg.height // 2), "DEMO MODE",
                  1.0, CYAN, 2, center=True)
            _text(panel, (self.left_w // 2, self.cfg.height // 2 + 34),
                  "no camera required", 0.6, DIM, 1, center=True)
            return

        view = cv2.flip(frame, 1) if self.cfg.mirror else frame
        for hand in hands:
            draw_hand_overlay(view, hand, flip_for_display=self.cfg.mirror)

        h, w = view.shape[:2]
        scale = min(self.left_w / w, self.cfg.height / h)
        resized = cv2.resize(view, (max(1, int(w * scale)), max(1, int(h * scale))))
        ox = (self.left_w - resized.shape[1]) // 2
        oy = (self.cfg.height - resized.shape[0]) // 2
        panel[:] = (18, 20, 24)
        panel[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
        cv2.line(panel, (self.left_w - 1, 0), (self.left_w - 1, self.cfg.height),
                 (52, 56, 64), 2)

    def _draw_robot(self, canvas, dt: float) -> None:
        prims: list = []
        for hand in self.hands:
            prims.extend(hand.solve())
        if self.follow and self.hands:
            focus = self.hands[0].position * 0.55
            self.cam_target += (focus - self.cam_target) * clamp(dt * 3.2, 0.0, 1.0)
        self.renderer.camera.target = self.cam_target.copy()
        self.renderer.camera._update()

        img = self.renderer.render(prims, grid=self.show_grid)
        canvas[:, self.left_w:] = img
        for hand in self.hands:
            self._draw_fingertip_labels(canvas, hand)

    def _draw_fingertip_labels(self, canvas, hand: RobotHand) -> None:
        cam = self.renderer.camera
        ss = self.renderer.ss
        for name, tip in hand.poses.items():
            u, v, d = cam.project_one(hand.world_point(tip))
            if d <= 0.02:
                continue
            px, py = int(self.left_w + u / ss), int(v / ss)
            if not (self.left_w < px < self.cfg.width and 0 < py < self.cfg.height):
                continue
            curl = float(np.clip(hand.flex(name), 0.0, 1.0))
            color = GREEN if curl < 0.35 else (YELLOW if curl < 0.7 else ORANGE)
            cv2.circle(canvas, (px, py), 3, color, -1, cv2.LINE_AA)

    def _draw_hud(self, canvas, hands, smoothed, dt: float) -> None:
        cfg = self.cfg
        g = self.gesture.reading

        # --- top strip: status -------------------------------------------
        canvas[0:54] = (26, 29, 34)
        _text(canvas, (16, 24), "ROBOHAND", 0.62, CYAN, 2)
        _text(canvas, (16, 42), "live hand -> robot hand simulation", 0.42, DIM, 1)

        if hands:
            label = g.label
            col = CYAN if g.name == "none" else GREEN
            _text(canvas, (cfg.width - 300, 30), label, 0.66, col, 2, right=True)
            if g.stable_for:
                _text(canvas, (cfg.width - 300, 47), f"held {g.stable_for} frames",
                      0.36, DIM, 1, right=True)
        else:
            _text(canvas, (cfg.width - 300, 40), "SHOW YOUR HAND TO THE CAMERA",
                  0.56, YELLOW, 2, right=True)

        fps = self.fps.tick(time.perf_counter())
        right = f"{fps:5.1f} fps   smoothing {self.cfg.smoothing:.2f}"
        if self.cfg.ros_urdf:
            if self.bridge is not None and self.bridge.alive:
                right += f"   ROS +{self.bridge.stats['frames_sent']}"
            else:
                right += "   ROS OFF"
        if self.preset_override is not None:
            right = f"PRESET   {right}"
        _text(canvas, (cfg.width - 16, 24), right, 0.48, DIM, 1, right=True)
        if self.demo is not None:
            _text(canvas, (cfg.width - 16, 42),
                  f"demo: {self.demo.stage} {self.demo.progress * 100:3.0f}%",
                  0.42, CYAN, 1, right=True)

        # --- bottom-left: per-finger curl bars ---------------------------
        if hands:
            self._draw_curl_panel(canvas, hands[0])

        # --- notices ------------------------------------------------------
        if self.too_dark and not hands:
            _text(canvas, (self.left_w // 2, self.cfg.height - 118),
                  "CAMERA FEED IS VERY DARK", 0.60, YELLOW, 2, center=True)
            _text(canvas, (self.left_w // 2, self.cfg.height - 94),
                  "add a lamp - the tracker needs light, not gain", 0.42, DIM, 1, center=True)
        if self.notice and time.time() < self.notice_until:
            _text(canvas, (cfg.width // 2, cfg.height - 74), self.notice,
                  0.6, YELLOW, 2, center=True)
        _text(canvas, (cfg.width // 2, cfg.height - 22),
              "H help   Q quit   S screenshot   1-7 preset poses   SPACE freeze   M mirror",
              0.42, DIM, 1, center=True)

    def _draw_curl_panel(self, canvas, hand: TrackedHand) -> None:
        x0, y0 = 18, self.cfg.height - 250
        _panel(canvas, x0 - 8, y0 - 30, 208, 190)
        _text(canvas, (x0, y0 - 12), f"FINGER JOINTS  ({hand.handedness})",
              0.42, CYAN, 1)
        for i, name in enumerate(FINGERS):
            y = y0 + i * 30
            curl = float(np.clip(hand.per_finger_curl.get(name, 0.0), 0.0, 1.0))
            col = GREEN if curl < 0.35 else (YELLOW if curl < 0.7 else ORANGE)
            _text(canvas, (x0, y + 12), FINGER_LABEL[name], 0.42, DIM, 1)
            _bar(canvas, x0 + 42, y, 120, 11, curl, col)
            _text(canvas, (x0 + 170, y + 12), f"{curl * 100:3.0f}", 0.40, col, 1)
        grip = hand.grip
        _text(canvas, (x0, y0 + 5 * 30 + 14), "GRIP", 0.42, DIM, 1)
        _bar(canvas, x0 + 42, y0 + 5 * 30 + 2, 120, 11, grip, ORANGE)
        _text(canvas, (x0 + 170, y0 + 5 * 30 + 14), f"{grip * 100:3.0f}", 0.40, ORANGE, 1)

    # -- input ---------------------------------------------------------------
    def _handle_key(self, key: int, canvas: np.ndarray) -> bool:
        if key in (255, -1):
            return self.running
        ch = chr(key).lower() if 32 <= key < 127 else ""
        if ch == "q" or key == 27:
            self.running = False
        elif ch == "h":
            self._toggle_help()
        elif ch == "s":
            self._screenshot(canvas)
        elif ch == " ":
            self.paused = not self.paused
        elif ch == "m":
            self.cfg.mirror = not self.cfg.mirror
            if self.tracker is not None:
                self.tracker.mirror = self.cfg.mirror
                self.tracker.close()
                self.tracker = HandTracker(
                    max_hands=self.cfg.max_hands,
                    model_complexity=self.cfg.model_complexity,
                    gain=self.cfg.gain, mirror=self.cfg.mirror,
                )
        elif ch == "g":
            self.show_grid = not self.show_grid
        elif ch == "c":
            self.follow = not self.follow
        elif ch == "[":
            self._set_smoothing(self.cfg.smoothing - 0.1)
        elif ch == "]":
            self._set_smoothing(self.cfg.smoothing + 0.1)
        elif ch in PRESETS:
            self.preset_override = dict(PRESETS[ch])
            self.notice = f"preset {ch}: {PRESET_NAMES[ch]}"
            self.notice_until = time.time() + 1.6
        elif ch == "0":
            self.preset_override = None
            self.notice = "preset cleared - live tracking"
            self.notice_until = time.time() + 1.6
        return self.running

    def _set_smoothing(self, value: float) -> None:
        self.cfg.smoothing = clamp(value, 0.0, 1.0)
        for sm in self.smoothers.values():
            sm.set_aggressiveness(self.cfg.smoothing)

    def _screenshot(self, canvas: np.ndarray) -> None:
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        self.screenshot_index += 1
        path = os.path.join(
            self.cfg.out_dir, f"robohand_{self.screenshot_index:03d}.png"
        )
        cv2.imwrite(path, canvas)
        self.notice = f"saved {path}"
        self.notice_until = time.time() + 1.8
        print(f"[shot] {path}")

    def _toggle_help(self) -> None:
        if getattr(self, "_help_win", None):
            cv2.destroyWindow("help")
            self._help_win = None
            return
        lines = [
            "ROBOHAND - controls",
            "",
            "  Q / ESC   quit",
            "  H         toggle this help",
            "  SPACE     freeze the simulation (tracking keeps running)",
            "  0         clear preset, follow the camera again",
            "  1..7      canned robot poses: open / half / fist /",
            "            yoke / pinch / flat / scissors",
            "  S         save a screenshot to ./captures",
            "  M         toggle mirrored camera view",
            "  G         toggle the 3D floor grid",
            "  C         toggle camera auto-follow",
            "  [ / ]     less / more input smoothing",
            "",
            "  Move your hand: the robot hand follows position and",
            "  wrist rotation. Curl your fingers: the robot joints flex.",
            "",
            "  Good lighting matters a lot - add a lamp if the feed",
            "  is dark, or run with --no-brightness to disable the",
            "  auto-exposure pass.",
        ]
        board = np.zeros((30 * len(lines) + 40, 720, 3), np.uint8)
        board[:] = (24, 27, 32)
        for i, line in enumerate(lines):
            col = CYAN if i == 0 else (DIM if line.startswith("  ") and not line.strip() else WHITE)
            _text(board, (24, 40 + i * 30), line, 0.52, col, 1)
        cv2.namedWindow("help", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("help", board)
        self._help_win = True


# --- small drawing helpers ---------------------------------------------------
def _text(img, org, s, scale, color, thick=1, right=False, center=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), base = cv2.getTextSize(s, font, scale, thick)
    x, y = org
    if center:
        x -= w // 2
    if right:
        x -= w
    cv2.putText(img, s, (x, y), font, scale, color, thick, cv2.LINE_AA)
    return w, h


def _panel(img, x, y, w, h):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 34, 40), -1)
    cv2.addWeighted(overlay, 0.86, img, 0.14, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (58, 64, 74), 1, cv2.LINE_AA)


def _bar(img, x, y, w, h, value, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 44, 52), -1)
    fill = int(round(w * float(np.clip(value, 0.0, 1.0))))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (74, 80, 92), 1, cv2.LINE_AA)


def _auto_brightness(frame: np.ndarray) -> np.ndarray:
    """Lift a dim webcam feed so MediaPipe can actually see the hand."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _blend_rot(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    v = a + (b - a) * float(np.clip(k, 0.0, 1.0))
    u, _, vt = np.linalg.svd(v)
    return u @ vt


# --- cli --------------------------------------------------------------------
def list_cameras() -> None:
    print("scanning for video devices...")
    found = False
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                print(f"  /dev/video{i}  ->  index {i}, {w}x{h}")
                found = True
        cap.release()
    if not found:
        print("  none found")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robohand",
        description="Mirror your hand from a webcam into a simulated 3D robot hand.",
    )
    p.add_argument("--source", default="0",
                   help="camera index, video file, or 'demo' (no camera)")
    p.add_argument("--demo", action="store_true", help="scripted motion, no camera")
    p.add_argument("--list-cameras", action="store_true", help="scan and exit")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--max-hands", type=int, default=2, choices=(1, 2))
    p.add_argument("--model-complexity", type=int, default=1, choices=(0, 1),
                   help="0 = faster, 1 = more accurate on hard poses")
    p.add_argument("--gain", type=float, default=1.35,
                   help="flexion gain; raise if the robot hand never fully closes")
    p.add_argument("--smoothing", type=float, default=0.45,
                   help="0 = raw and jittery, 1 = very smooth and laggy")
    p.add_argument("--move-gain", type=float, default=0.30,
                   help="how far the robot hand travels for hand movement (m)")
    p.add_argument("--rotate", type=float, default=1.0,
                   help="wrist-rotation follow, 0 disables")
    p.add_argument("--no-brightness", action="store_true",
                   help="disable the auto-exposure pass on dark feeds")
    p.add_argument("--no-mirror", dest="mirror", action="store_false",
                   help="do not mirror the camera view")
    p.add_argument("--record", default="", help="write the window to an .mp4")
    p.add_argument("--ros", metavar="URDF", default="",
                   help="stream the tracked hand to a ROS 2 simulator via this URDF")
    p.add_argument("--ros-hand", default="Right", choices=("Right", "Left"),
                   help="which hand model to drive")
    p.add_argument("--ros-workspace", default="",
                   help="ros2_ws to use (default: the bundled one)")
    p.add_argument("--export-urdf", metavar="PATH", default="",
                   help="write the rig as URDF to PATH and exit "
                        "(use '-' for stdout); --hand picks the handedness")
    p.add_argument("--hand", default="Right", choices=("Right", "Left"),
                   help="handedness for --export-urdf")
    p.add_argument("--export-joint-angles", action="store_true",
                   help="with --export-urdf, also print the joint table")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_cameras:
        list_cameras()
        return 0

    if args.export_urdf:
        from .urdf_export import build_urdf, joint_table

        xml = '<?xml version="1.0"?>\n' + build_urdf(args.hand)
        if args.export_urdf == "-":
            sys.stdout.write(xml)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.export_urdf)),
                        exist_ok=True)
            with open(args.export_urdf, "w") as fh:
                fh.write(xml)
            print(f"wrote {args.export_urdf}  ({args.hand} hand)")
        if args.export_joint_angles:
            print(f"\n{'joint':<14}{'min':>9}{'max':>9}")
            for name, lo, hi, _default in joint_table():
                print(f"{name:<14}{lo:>9.4f}{hi:>9.4f}")
        return 0

    src = args.source
    demo = args.demo or (isinstance(src, str) and src.lower() == "demo")
    source: int | str = 0
    if not demo:
        source = int(src) if str(src).isdigit() else src

    cfg = Config(
        source=source,
        width=args.width,
        height=args.height,
        demo=demo,
        mirror=args.mirror,
        no_brightness=args.no_brightness,
        gain=args.gain,
        smoothing=args.smoothing,
        move_gain=args.move_gain,
        rotate=args.rotate,
        max_hands=args.max_hands,
        model_complexity=args.model_complexity,
        record=args.record,
        ros_urdf=args.ros,
        ros_hand=args.ros_hand,
        ros_workspace=args.ros_workspace,
    )
    app = RoboHandApp(cfg)
    cv2.namedWindow(cfg.window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.window, cfg.width, cfg.height)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
