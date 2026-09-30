"""Anthropomorphic 5-finger robot hand driven by forward kinematics.

No mesh files, no OpenGL: the hand is a small articulated rig (a palm shell,
a wrist, a knuckle bar, and 5 x 3 revolute joints) evaluated analytically.
A `signal` dict from :mod:`robohand.hand_tracker` sets every joint angle, and
`solve()` returns render-ready primitives in world space.

Local frame convention (per finger):
    +Y  along the bone, pointing away from the wrist
    +Z  palm normal (points *out of* the palm, i.e. world -Z)
    +X  transverse axis; positive rotation about it = flexion
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .math3d import (
    IDENTITY,
    cross,
    dot,
    frame_from_axes,
    norm,
    rot_axis,
    rot_z,
    vec,
)

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

PALM_NORMAL = vec(0.0, 0.0, -1.0)
_X_LOCAL = vec(1.0, 0.0, 0.0)

# --- rig geometry (metres) ---------------------------------------------------
PALM_CENTER = vec(0.0, -0.046, 0.0)
PALM_HALF = vec(0.036, 0.049, 0.0145)
KNUCKLE_BAR_RADIUS = 0.0098

# name -> (base position, splay about palm normal, [(len, radius, max_flex)])
FINGER_RIG: dict[str, tuple] = {
    "index": (vec(-0.0235, 0.0, 0.0), math.radians(-9.0),
              [(0.043, 0.0112, math.radians(95)),
               (0.026, 0.0094, math.radians(105)),
               (0.019, 0.0080, math.radians(65))]),
    "middle": (vec(-0.0070, 0.0020, 0.0), math.radians(-2.0),
               [(0.049, 0.0115, math.radians(95)),
                (0.031, 0.0096, math.radians(105)),
                (0.021, 0.0082, math.radians(65))]),
    "ring": (vec(0.0100, 0.0010, 0.0), math.radians(5.0),
             [(0.045, 0.0110, math.radians(95)),
              (0.028, 0.0092, math.radians(105)),
              (0.020, 0.0079, math.radians(65))]),
    "pinky": (vec(0.0255, -0.0040, 0.0), math.radians(13.0),
              [(0.035, 0.0096, math.radians(95)),
               (0.021, 0.0082, math.radians(100)),
               (0.017, 0.0072, math.radians(65))]),
    "thumb": (vec(-0.0300, -0.0520, 0.004), None,
              [(0.032, 0.0126, math.radians(42)),
               (0.030, 0.0111, math.radians(46)),
               (0.024, 0.0095, math.radians(62))]),
}

THUMB_REST_DIR = norm(vec(-0.72, -0.66, 0.22))
THUMB_OPPOSITION_MAX = math.radians(62.0)
THUMB_ABDUCTION_MAX = math.radians(26.0)

# --- palette (BGR) -----------------------------------------------------------
C_SHELL = (122, 130, 146)
C_SHELL_DARK = (52, 57, 66)
C_BONE_OPEN = (150, 156, 168)
C_BONE_SHUT = (86, 132, 214)
C_JOINT = (64, 70, 82)
C_TIP = (196, 202, 214)
C_WRIST = (78, 84, 98)
C_RAIL = (136, 144, 158)


def _blend(a: tuple, b: tuple, t: float) -> tuple:
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


@dataclass
class Primitive:
    """Render-ready shape: a tapered capsule or an oriented box."""

    kind: str                      # "capsule" | "box"
    color: tuple
    p0: np.ndarray | None = None
    p1: np.ndarray | None = None
    r0: float = 0.0
    r1: float = 0.0
    center: np.ndarray | None = None
    half: np.ndarray | None = None
    rot: np.ndarray | None = None
    metal: float = 0.85            # 0 = matte, 1 = strong specular band
    edge: float = 1.0              # outline strength
    bias: float = 0.0              # depth-sort nudge, metres (+ = drawn on top)
    tag: str = ""


@dataclass
class RobotHand:
    """A single robot hand; call `set_pose()` then `solve()`."""

    handedness: str = "Right"
    position: np.ndarray = field(default_factory=lambda: vec(0.0, 0.0, 0.0))
    spread: float = 0.0            # 0 = fingers together, 1 = splayed
    world_rot: np.ndarray | None = None   # maps the local rig frame to world
    _signal: dict = field(default_factory=dict)
    joints: dict[str, list[np.ndarray]] = field(default_factory=dict)
    poses: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        self._mirror = -1.0 if self.handedness.lower().startswith("l") else 1.0

    # -- control ------------------------------------------------------------
    def set_pose(self, signal: dict) -> None:
        self._signal = signal

    def flex(self, finger: str) -> float:
        """Current commanded flexion for one finger, 0 = straight, 1 = fist."""
        return float(self._signal.get(f"{finger}_flex", 0.0))

    def set_world_rotation(self, rot) -> None:
        """Set the rig's orientation in world space (3x3 column-major basis)."""
        if rot is None:
            self.world_rot = None
        else:
            self.world_rot = np.asarray(rot, dtype=float).reshape(3, 3)

    # -- forward kinematics --------------------------------------------------
    def _finger_base_frame(self, name: str, signal: dict) -> tuple[np.ndarray, np.ndarray]:
        base, splay, _ = FINGER_RIG[name]
        if name == "thumb":
            opposition = float(signal.get("thumb_opposition", 0.0))
            abduction = float(signal.get("thumb_abduction", 0.0))
            # Sweep the thumb across the palm (positive about the palm normal),
            # then lift it clear of the palm plane when the hand is open.
            y_dir = rot_z(+opposition * THUMB_OPPOSITION_MAX) @ THUMB_REST_DIR
            y_dir = rot_axis(vec(0.0, 1.0, 0.0), -abduction * THUMB_ABDUCTION_MAX) @ y_dir
        else:
            sp = splay * (1.0 + 0.55 * self.spread)
            y_dir = rot_z(sp) @ vec(0.0, 1.0, 0.0)

        # Project the palm normal into the plane perpendicular to the bone.
        # (Adding the two vectors instead of orthogonalising would tilt the
        # whole chain out of the palm plane by ~45 deg.)
        y_dir = norm(y_dir)
        z_dir = norm(PALM_NORMAL - dot(PALM_NORMAL, y_dir) * y_dir)
        y_dir = norm(y_dir - dot(y_dir, z_dir) * z_dir)
        return base.copy(), frame_from_axes(cross(y_dir, z_dir), y_dir, z_dir)

    def chain(self, name: str, signal: dict | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
        """Forward kinematics for one finger.

        Returns 4 `(position, rotation)` pairs - one per joint. `rotation` has
        the bone direction as its local +Y and the flexion axis as its local
        +X, which is exactly the frame convention the renderer and the URDF
        exporter both rely on.
        """
        sig = signal if signal is not None else (self._signal or {})
        _, _, segs = FINGER_RIG[name]
        flex = float(np.clip(sig.get(f"{name}_flex", 0.0), 0.0, 1.25))
        pos, rot = self._finger_base_frame(name, sig)
        frames = [(pos.copy(), rot.copy())]
        for length, _radius, max_flex in segs:
            rot = rot @ rot_axis(_X_LOCAL, flex * max_flex)
            pos = pos + rot @ vec(0.0, length, 0.0)
            frames.append((pos.copy(), rot.copy()))
        return frames

    def solve(self) -> list[Primitive]:
        """Evaluate the rig; returns primitives in the hand's local frame."""
        sig = self._signal or {}
        prims: list[Primitive] = []
        self.joints = {}
        self.poses = {}

        shell = _blend(C_SHELL, C_SHELL_DARK, 0.15)
        prims.append(Primitive(
            kind="box", color=shell, center=PALM_CENTER, half=PALM_HALF, rot=IDENTITY,
            metal=0.55, tag="palm",
        ))
        # Back plate + wrist mount: makes the palm read as a machined shell.
        prims.append(Primitive(
            kind="box", color=(124, 132, 148),
            center=vec(0.0, -0.038, 0.0155), half=vec(0.027, 0.040, 0.0035),
            rot=IDENTITY, metal=0.5, bias=0.0008, tag="backplate",
        ))
        prims.append(Primitive(
            kind="capsule", color=C_WRIST, p0=vec(0.0, -0.080, 0.0),
            p1=vec(0.0, -0.130, 0.0), r0=0.027, r1=0.022, metal=0.7, tag="wrist",
        ))
        prims.append(Primitive(
            kind="capsule", color=(44, 48, 58), p0=vec(0.0, -0.126, 0.0),
            p1=vec(0.0, -0.140, 0.0), r0=0.029, r1=0.029, metal=0.4, edge=0.8,
            bias=0.0012, tag="wrist_collar",
        ))
        # Thenar bulge - visually roots the thumb into the palm.
        prims.append(Primitive(
            kind="capsule", color=(104, 111, 126),
            p0=vec(-0.008, -0.026, 0.000), p1=vec(-0.038, -0.058, 0.003),
            r0=0.014, r1=0.012, metal=0.6, bias=0.0004, tag="thenar",
        ))
        prims.append(Primitive(
            kind="capsule", color=C_RAIL, p0=vec(-0.027, 0.001, 0.010),
            p1=vec(0.027, 0.001, 0.010), r0=KNUCKLE_BAR_RADIUS,
            r1=KNUCKLE_BAR_RADIUS, metal=0.9, bias=0.001, tag="knuckle_bar",
        ))

        for name in FINGERS:
            base, _, segs = FINGER_RIG[name]
            flex = float(np.clip(sig.get(f"{name}_flex", 0.0), 0.0, 1.25))
            frames = self.chain(name, sig)
            base_frame = frames[0][1].copy()
            pts = [p.copy() for p, _ in frames]

            # Actuator housing bolted to the back of each knuckle.
            if name != "thumb":
                prims.append(Primitive(
                    kind="box", color=(112, 120, 136),
                    center=pts[0] - base_frame[:, 2] * 0.0115,
                    half=vec(0.0125, 0.0110, 0.0085),
                    rot=base_frame, metal=0.5, bias=0.0006, tag=f"{name}_servo",
                ))

            self.joints[name] = pts
            body = _blend(C_BONE_OPEN, C_BONE_SHUT, flex)

            for i, p in enumerate(pts[:-1]):
                seg_radius = segs[i][1]
                prims.append(Primitive(
                    kind="capsule", color=C_JOINT, p0=p, p1=p + vec(0.0, 1e-4, 0.0),
                    r0=seg_radius * 0.60, r1=seg_radius * 0.60,
                    metal=0.30, edge=0.8, bias=0.0035, tag=f"{name}_hinge{i}",
                ))
            tip_radius = segs[-1][1] * 0.86
            prims.append(Primitive(
                kind="capsule", color=C_TIP, p0=pts[-1], p1=pts[-1] + vec(0.0, 1e-4, 0.0),
                r0=tip_radius * 0.92, r1=tip_radius * 0.92,
                metal=0.5, bias=0.0035, tag=f"{name}_tip",
            ))

            for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
                (length, radius, _mf) = segs[i]
                prims.append(Primitive(
                    kind="capsule", color=body, p0=a, p1=b,
                    r0=radius, r1=radius * 0.88, metal=0.95, tag=f"{name}{i}",
                ))
            self.poses[name] = pts[-1].copy()

        if self._mirror < 0:
            prims = _mirror_primitives(prims, self._mirror)
            for key in self.joints:
                self.joints[key] = [p * self._mirror for p in self.joints[key]]
            for key in self.poses:
                self.poses[key] = self.poses[key] * self._mirror
        if self.world_rot is not None:
            prims = _rotate_primitives(prims, self.world_rot)
        prims = _translate_primitives(prims, self.position)
        return prims

    def world_point(self, local_point) -> np.ndarray:
        """Map a point from the rig's local frame into world space."""
        p = np.asarray(local_point, dtype=float)
        if self.world_rot is not None:
            p = self.world_rot @ p
        return p + self.position

    def mirror_point(self, p) -> np.ndarray:
        """Reflect a local point for a left hand (chain() is always right-hand)."""
        return np.asarray(p, dtype=float) * np.array([self._mirror, 1.0, 1.0])

    def mirror_rotation(self, rot) -> np.ndarray:
        """Reflect a local frame for a left hand, keeping it a *proper*
        rotation.

        A plain `diag(-1,1,1) @ R` has determinant -1, i.e. it is a reflection,
        which no rotation matrix (and therefore no URDF `rpy`) can express. The
        third axis is therefore also negated, which restores det = +1. Nothing
        depends on that axis: the bone runs along local +Y and flexion is about
        local +X, and both are reflected correctly.
        """
        R = np.asarray(rot, dtype=float)
        if self._mirror > 0:
            return R.copy()
        return np.diag([-1.0, 1.0, -1.0]) @ R


def _mirror_primitives(prims: list[Primitive], sign: float) -> list[Primitive]:
    """Reflect across the YZ plane. Box frames need mirroring too, otherwise a
    left hand's shells come out with the wrong orientation."""
    m = np.array([sign, 1.0, 1.0])
    out = []
    for p in prims:
        q = Primitive(**{**p.__dict__})
        for attr in ("p0", "p1", "center"):
            v = getattr(q, attr)
            if v is not None:
                setattr(q, attr, np.asarray(v, float) * m)
        if q.rot is not None:
            q.rot = np.diag(m) @ np.asarray(q.rot, float)
        out.append(q)
    return out


def _rotate_primitives(prims: list[Primitive], rot: np.ndarray) -> list[Primitive]:
    out = []
    for p in prims:
        q = Primitive(**{**p.__dict__})
        for attr in ("p0", "p1", "center"):
            v = getattr(q, attr)
            if v is not None:
                setattr(q, attr, rot @ np.asarray(v, float))
        if q.rot is not None:
            q.rot = rot @ np.asarray(q.rot, float)
        out.append(q)
    return out


def _translate_primitives(prims: list[Primitive], offset: np.ndarray) -> list[Primitive]:
    if not np.any(offset):
        return prims
    out = []
    for p in prims:
        q = Primitive(**{**p.__dict__})
        for attr in ("p0", "p1", "center"):
            v = getattr(q, attr)
            if v is not None:
                setattr(q, attr, np.asarray(v, float) + offset)
        out.append(q)
    return out
