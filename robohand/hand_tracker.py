"""Camera hand tracking: MediaPipe 21 landmarks -> robot joint targets.

Pipeline per frame:
    BGR frame -> RGB -> MediaPipe Hands -> landmarks
    landmarks -> per-finger flexion angles + thumb opposition + palm pose
    -> `HandSignal` (a flat dict of named scalars, easy to smooth / log / map)

The `HandSignal` dict is deliberately flat and primitive-only so it can be fed
to a smoother, a serial port, ROS, or a simulator without further glue.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from .math3d import cross

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("opencv is required: pip install opencv-contrib-python") from exc

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    raise SystemExit("mediapipe is required: pip install mediapipe") from exc


# --- MediaPipe hand landmark indices -----------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# (mcp, pip, dip, tip) per finger
FINGER_JOINTS: dict[str, tuple[int, int, int, int]] = {
    "thumb": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

# Curl calibration: joint angle (deg) at which a finger reads as open / closed.
ANGLE_OPEN = 168.0
ANGLE_CLOSED = 42.0

SIGNAL_KEYS: tuple[str, ...] = (
    "thumb_flex", "index_flex", "middle_flex", "ring_flex", "pinky_flex",
    "thumb_opposition", "palm_yaw", "palm_pitch", "palm_roll",
    "wrist_flex", "wrist_ulnar", "x", "y", "z", "grip", "openness",
)


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 == edge0:
        return 0.0
    t = float(np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at b (degrees)."""
    v1, v2 = a - b, c - b
    n = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if n < 1e-9:
        return 180.0
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


@dataclass
class TrackedHand:
    """One detected hand for one frame."""

    present: bool = False
    handedness: str = "Right"
    handedness_score: float = 0.0
    #: 21x3 normalised image coords (x, y in [0,1]; z relative depth).
    landmarks: np.ndarray | None = None
    #: 21x3 metric coords in metres, hand-centred, from MediaPipe.
    world: np.ndarray | None = None
    signal: dict = field(default_factory=dict)
    per_finger_curl: dict[str, float] = field(default_factory=dict)
    #: 3x3, columns = palm x (index->pinky), y (wrist->fingers), z (out of the
    #: back of the hand), already mapped into the renderer's coordinate frame.
    palm_rot: np.ndarray | None = None

    @property
    def openness(self) -> float:
        return float(self.signal.get("openness", 0.0))

    @property
    def grip(self) -> float:
        return float(self.signal.get("grip", 0.0))


class HandTracker:
    """Wraps MediaPipe Hands and converts frames into `TrackedHand` results."""

    def __init__(
        self,
        max_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        gain: float = 1.35,
        mirror: bool = True,
    ) -> None:
        # MediaPipe is chatty on stderr; keep the console readable.
        os.environ.setdefault("GLOG_minloglevel", "2")
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.gain = float(gain)
        self.mirror = bool(mirror)
        self._ready = False

    # -- lifecycle ------------------------------------------------------------
    def close(self) -> None:
        if self._ready:
            self._hands.close()
            self._ready = False

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- main entry point -----------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> list[TrackedHand]:
        """Detect hands in a BGR frame. Returns [] when nothing is found."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        view = cv2.flip(frame_bgr, 1) if self.mirror else frame_bgr
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        self._ready = True

        out: list[TrackedHand] = []
        landmarks = getattr(res, "multi_hand_landmarks", None)
        worlds = getattr(res, "multi_hand_world_landmarks", None)
        handedness = getattr(res, "multi_handedness", None)
        if not landmarks:
            return out

        for i, lm in enumerate(landmarks):
            world = None
            if worlds is not None and i < len(worlds):
                world = np.array([[p.x, p.y, p.z] for p in worlds[i].landmark], float)
            norm_lm = np.array([[p.x, p.y, p.z] for p in lm.landmark], float)

            label, score = "Right", 0.0
            if handedness is not None and i < len(handedness):
                cls = handedness[i].classification[0]
                label, score = str(cls.label), float(cls.score)

            hand = TrackedHand(
                present=True,
                handedness=label,
                handedness_score=score,
                landmarks=norm_lm,
                world=world,
            )
            self._measure(hand, world if world is not None else norm_lm)
            out.append(hand)
        return out

    # -- signal extraction ----------------------------------------------------
    def _measure(self, hand: TrackedHand, pts: np.ndarray) -> None:
        """Derive joint angles + palm orientation from landmark coordinates."""
        if pts is None or pts.shape[0] < 21:
            hand.signal = {k: 0.0 for k in SIGNAL_KEYS}
            return

        wrist = pts[WRIST]
        middle_mcp = pts[MIDDLE_MCP]
        hand_scale = float(np.linalg.norm(middle_mcp - wrist))
        if hand_scale < 1e-6:
            hand_scale = 1.0

        curls: dict[str, float] = {}
        for name, (mcp, pip, dip, tip) in FINGER_JOINTS.items():
            curls[name] = self._finger_curl(pts, (mcp, pip, dip, tip), hand_scale)
        hand.per_finger_curl = curls

        signal: dict[str, float] = {}
        for name in FINGERS:
            signal[f"{name}_flex"] = float(np.clip(curls[name] * self.gain, 0.0, 1.25))

        # Thumb opposition: how far the tip has swung across the palm.
        palm_n = np.cross(pts[INDEX_MCP] - wrist, pts[PINKY_MCP] - wrist)
        across = float(np.dot(pts[THUMB_TIP] - wrist, palm_n))
        side = pts[THUMB_TIP] - pts[INDEX_MCP]
        opposed = across > 0.0 and float(np.dot(side, palm_n)) < 0.0
        signal["thumb_opposition"] = 1.0 if opposed else 0.0
        # Blend in a continuous term so the thumb sweeps instead of snapping.
        sweep = float(np.clip((across / hand_scale + 0.25) / 1.1, 0.0, 1.0))
        signal["thumb_opposition"] = float(np.clip(0.65 * sweep + 0.35 * opposed, 0.0, 1.0))

        # Palm orientation -> basis for the robot rig, plus euler readouts.
        # ex/ey are not perpendicular on a real hand, so the basis is
        # orthonormalised: a raw matrix would skew and scale the rig.
        local = self._basis(pts[PINKY_MCP] - pts[INDEX_MCP],
                            pts[MIDDLE_MCP] - wrist)
        ex, ey, ez = local[:, 0], local[:, 1], local[:, 2]
        # Image space is y-down / z-away; the renderer is y-up / z-toward-viewer.
        hand.palm_rot = _IMAGE_TO_WORLD @ local

        signal["palm_yaw"] = math.atan2(float(np.dot(ez, _X)), float(np.dot(ez, _Y)))
        signal["palm_pitch"] = math.asin(float(np.clip(np.dot(ez, -_Z), -1.0, 1.0)))
        signal["palm_roll"] = math.atan2(float(np.dot(ey, _Y)), float(np.dot(ey, _X)))

        # Wrist articulation, from the hand's own frame.
        wx, wy, wz = self._to_hand_frame(pts, ex, ey, ez)
        signal["wrist_flex"] = float(np.clip(math.atan2(-wz, max(wy, 1e-6)), -1.2, 1.2))
        signal["wrist_ulnar"] = float(np.clip(math.atan2(wx, max(wy, 1e-6)), -1.2, 1.2))

        signal["x"], signal["y"], signal["z"] = (float(v) for v in wrist)

        four = [curls[f] for f in ("index", "middle", "ring", "pinky")]
        signal["grip"] = float(np.clip(np.mean(four), 0.0, 1.0))
        signal["openness"] = float(np.clip(1.0 - signal["grip"], 0.0, 1.0))
        hand.signal = signal

    def _finger_curl(
        self, pts: np.ndarray, joints: tuple[int, int, int, int], scale: float
    ) -> float:
        """Combine joint angles with tip-to-knuckle distance for a stable curl."""
        mcp, pip, dip, tip = (pts[i] for i in joints)

        a_pip = _angle_at(mcp, pip, dip)
        a_dip = _angle_at(pip, dip, tip)
        ang = 0.62 * a_pip + 0.38 * a_dip
        curl_ang = np.clip((ANGLE_OPEN - ang) / (ANGLE_OPEN - ANGLE_CLOSED), 0.0, 1.0)

        chain = (
            float(np.linalg.norm(pip - mcp))
            + float(np.linalg.norm(dip - pip))
            + float(np.linalg.norm(tip - dip))
        )
        ratio = float(np.linalg.norm(tip - mcp)) / max(chain, 1e-6)
        curl_dist = np.clip((0.97 - ratio) / (0.97 - 0.52), 0.0, 1.0)

        return float(np.clip(0.68 * curl_ang + 0.32 * curl_dist, 0.0, 1.0))

    @staticmethod
    def _unit(v: np.ndarray, fallback=(0.0, 1.0, 0.0)) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else np.asarray(fallback, dtype=float)

    @classmethod
    def _basis(cls, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Orthonormal (ex, ey, ez) from two arbitrary directions.

        Gram-Schmidt, with a guaranteed-valid fallback when the inputs are
        degenerate or parallel - otherwise a bad frame yields a singular
        rotation that would shear the robot hand.
        """
        ex = cls._unit(v1, fallback=(1.0, 0.0, 0.0))
        ey = np.asarray(v2, dtype=float) - float(np.dot(v2, ex)) * ex
        if float(np.linalg.norm(ey)) < 1e-6:
            for cand in ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)):
                if abs(float(np.dot(np.asarray(cand), ex))) < 0.9:
                    ey = np.asarray(cand, dtype=float)
                    break
            else:
                ey = np.array([0.0, 1.0, 0.0])
        ey = cls._unit(ey, fallback=(0.0, 1.0, 0.0))
        return np.column_stack([ex, ey, cross(ex, ey)])

    @staticmethod
    def _to_hand_frame(
        pts: np.ndarray, nx: np.ndarray, ny: np.ndarray, nz: np.ndarray
    ) -> tuple[float, float, float]:
        """Express (middle_mcp - wrist) in the palm's own basis."""
        v = pts[MIDDLE_MCP] - pts[WRIST]
        return (
            float(np.dot(v, nx)),
            float(np.dot(v, ny)),
            float(np.dot(v, nz)),
        )


_X = np.array([1.0, 0.0, 0.0])
_Y = np.array([0.0, 1.0, 0.0])
_Z = np.array([0.0, 0.0, 1.0])

# MediaPipe image space is (x right, y down, z away from the lens); the renderer
# uses (x right, y up, z toward the viewer). This is a 180 deg turn about X, so
# it is a proper rotation and preserves handedness.
_IMAGE_TO_WORLD = np.diag([1.0, -1.0, -1.0])


def draw_hand_overlay(
    canvas: np.ndarray,
    hand: TrackedHand,
    flip_for_display: bool = True,
) -> None:
    """Draw skeleton + per-finger curl bars onto a camera frame."""
    if not hand.present or hand.landmarks is None:
        return
    h, w = canvas.shape[:2]
    pts = hand.landmarks
    xy = pts[:, :2].copy()
    if flip_for_display:
        xy[:, 0] = 1.0 - xy[:, 0]
    px = (xy * np.array([w, h])).astype(int)

    for a, b in HAND_CONNECTIONS:
        ca = np.clip(hand.per_finger_curl.get(FINGER_CHAIN_NAME.get(a, ""), 0.5), 0, 1)
        cb = np.clip(hand.per_finger_curl.get(FINGER_CHAIN_NAME.get(b, ""), 0.5), 0, 1)
        c = 0.5 * (ca + cb)
        color = _BGR_OPEN if c < 0.5 else _BGR_CLOSED
        cv2.line(canvas, tuple(px[a]), tuple(px[b]), color, 3, cv2.LINE_AA)

    for i, (x, y) in enumerate(px):
        name = FINGER_CHAIN_NAME.get(i, "")
        c = hand.per_finger_curl.get(name, 0.0) if name else 0.0
        color = _BGR_OPEN if c < 0.5 else _BGR_CLOSED
        r = 5 if i in (WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP) else 3
        cv2.circle(canvas, (int(x), int(y)), r, color, -1, cv2.LINE_AA)

    tip = hand.landmarks[THUMB_TIP][:2]
    index_tip = hand.landmarks[INDEX_TIP][:2]
    dist_px = float(np.linalg.norm((tip - index_tip) * np.array([w, h])))
    pinch = 1.0 - float(np.clip(dist_px / (0.32 * w), 0.0, 1.0))
    if pinch > 0.65:
        c = tuple(int(_BGR_OPEN[i] * pinch + _BGR_WARN[i] * (1 - pinch)) for i in range(3))
        cv2.circle(canvas, (int(px[THUMB_TIP][0]), int(px[THUMB_TIP][1])),
                   int(9 + 12 * pinch), c, 2, cv2.LINE_AA)


FINGER_CHAIN_NAME: dict[int, str] = {}
for _name, _j in FINGER_JOINTS.items():
    for _idx in _j:
        FINGER_CHAIN_NAME[_idx] = _name
FINGER_CHAIN_NAME[WRIST] = "palm"

_BGR_OPEN = (86, 214, 122)      # green - finger extended
_BGR_CLOSED = (72, 96, 240)     # orange/red - finger curled
_BGR_WARN = (60, 200, 250)      # yellow - pinch
