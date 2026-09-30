"""robohand - mirror your hand into a simulated 5-finger robot hand."""

from __future__ import annotations

__version__ = "1.0.0"

from .math3d import Vec3, clamp, norm, vec
from .robot_hand import FINGERS, Primitive, RobotHand
from .renderer import Camera3D, Renderer
from .smoothing import FpsMeter, OneEuroFilter, PoseSmoother

__all__ = [
    "FINGERS",
    "Camera3D",
    "FpsMeter",
    "OneEuroFilter",
    "PoseSmoother",
    "Primitive",
    "Renderer",
    "RobotHand",
    "Vec3",
    "clamp",
    "norm",
    "vec",
    "__version__",
]
