"""Minimal 3D math helpers (rotation matrices, vector ops) built on numpy.

Right-handed coordinate convention used by the whole project:
    +X  right
    +Y  up (fingers point up)
    +Z  toward the viewer
Palm normal is -Z, so a positive "flexion" curls fingertips away from the viewer.
"""

from __future__ import annotations

import math

import numpy as np

Vec3 = np.ndarray
Mat3 = np.ndarray

IDENTITY = np.eye(3, dtype=float)


def vec(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vec3:
    return np.array([x, y, z], dtype=float)


def norm(v: Vec3, eps: float = 1e-9) -> Vec3:
    return v / (float(np.linalg.norm(v)) + eps)


def cross(a: Vec3, b: Vec3) -> Vec3:
    return np.cross(a, b)


def dot(a: Vec3, b: Vec3) -> float:
    return float(np.dot(a, b))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


def rot_x(angle: float) -> Mat3:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(angle: float) -> Mat3:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(angle: float) -> Mat3:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def skew(v: Vec3) -> Mat3:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float
    )


def rot_axis(axis: Vec3, angle: float) -> Mat3:
    """Rodrigues rotation about an arbitrary axis."""
    k = norm(axis)
    s, c = math.sin(angle), math.cos(angle)
    return np.eye(3) + s * skew(k) + (1.0 - c) * (skew(k) @ skew(k))


def rot_between(a: Vec3, b: Vec3) -> Mat3:
    """Shortest-arc rotation taking unit vector a onto unit vector b."""
    a, b = norm(a), norm(b)
    v = cross(a, b)
    d = dot(a, b)
    if float(np.linalg.norm(v)) < 1e-9:
        return np.eye(3) if d > 0 else rot_axis(
            norm(np.cross(a, np.array([1.0, 0.0, 0.0]) + 1e-6)), math.pi
        )
    return rot_axis(v, math.atan2(float(np.linalg.norm(v)), d))


def frame_from_axes(x: Vec3, y: Vec3, z: Vec3) -> Mat3:
    """Build a right-handed frame; columns are the given axes (Gram-Schmidt)."""
    z = norm(z)
    x = norm(x - dot(x, z) * z)
    y = cross(z, x)
    return np.column_stack([x, y, z])


def euler_zyx(yaw: float, pitch: float, roll: float) -> Mat3:
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def mat_to_euler_zyx(m: Mat3) -> tuple[float, float, float]:
    """Inverse of :func:`euler_zyx` (handles gimbal lock)."""
    sy = float(np.clip(m[2, 0], -1.0, 1.0))
    pitch = math.asin(-sy)
    if abs(sy) > 0.9999:
        yaw = math.atan2(-m[0, 1], m[1, 1])
        roll = 0.0
    else:
        yaw = math.atan2(m[1, 0], m[0, 0])
        roll = math.atan2(m[2, 1], m[2, 2])
    return yaw, pitch, roll
