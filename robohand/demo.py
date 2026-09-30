"""Scripted hand motion for `--demo` mode.

Lets the simulation be exercised (and screenshotted) with no camera, and gives
a known-good reference for tuning the rig and the renderer.
"""

from __future__ import annotations

import math

import numpy as np

from .robot_hand import FINGERS

# (name, duration s, curl targets) - curls are eased, so transitions are smooth.
SCRIPT: list[tuple[str, float, dict[str, float]]] = [
    ("open palm", 2.2, {f: 0.04 for f in FINGERS}),
    ("closing", 1.6, {f: 1.00 for f in FINGERS}),
    ("fist", 1.8, {f: 1.00 for f in FINGERS}),
    ("opening", 1.6, {f: 0.04 for f in FINGERS}),
    ("point", 2.0, {"thumb": 0.95, "index": 0.05, "middle": 1.0, "ring": 1.0, "pinky": 1.0}),
    ("peace", 2.0, {"thumb": 0.90, "index": 0.05, "middle": 0.05, "ring": 1.0, "pinky": 1.0}),
    ("pinch", 2.0, {"thumb": 0.70, "index": 0.55, "middle": 0.20, "ring": 0.18, "pinky": 0.18}),
    ("splayed", 2.0, {f: 0.08 for f in FINGERS}),
]

_TOTAL = sum(dur for _, dur, _ in SCRIPT)


def _ease(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


class DemoDriver:
    """Generates a plausible `HandSignal` dict over time."""

    def __init__(self) -> None:
        self.t = 0.0
        self.stage = ""
        self.progress = 0.0

    def step(self, dt: float) -> dict:
        self.t = (self.t + dt) % _TOTAL
        acc = 0.0
        idx = 0
        for i, (name, dur, curls) in enumerate(SCRIPT):
            if self.t < acc + dur:
                idx = i
                break
            acc += dur
        name, dur, curls = SCRIPT[idx]
        local = (self.t - acc) / max(dur, 1e-6)
        self.stage, self.progress = name, local

        # Arrive at this stage's pose over the first ~28% of the window, hold it,
        # then glide toward the next stage over the last ~28%.
        prev = SCRIPT[(idx - 1) % len(SCRIPT)][2]
        nxt = SCRIPT[(idx + 1) % len(SCRIPT)][2]
        if local < 0.28:
            a, b, frac = prev, curls, _ease(local / 0.28)
        elif local < 0.72:
            a, b, frac = curls, curls, 0.0
        else:
            a, b, frac = curls, nxt, _ease((local - 0.72) / 0.28)

        signal: dict = {}
        for f in FINGERS:
            signal[f"{f}_flex"] = float(np.clip(a[f] + (b[f] - a[f]) * frac, 0.0, 1.0))

        signal["thumb_opposition"] = float(
            np.clip(0.15 + 0.85 * signal["index_flex"], 0.0, 1.0)
        )
        signal["grip"] = float(np.mean([signal[f"{f}_flex"] for f in
                                        ("index", "middle", "ring", "pinky")]))
        signal["openness"] = 1.0 - signal["grip"]

        # Gentle lateral sweep + wrist rotation so the 3D motion is visible.
        sweep = math.sin(self.t * 0.55)
        signal["x"] = 0.5 + 0.16 * sweep
        signal["y"] = 0.5 + 0.07 * math.cos(self.t * 0.37)
        a = 0.30 * math.sin(self.t * 0.42)
        signal["palm_yaw"] = a
        signal["palm_pitch"] = 0.20 * math.cos(self.t * 0.31)
        signal["palm_roll"] = 0.16 * math.sin(self.t * 0.27)
        return signal

    def palm_rotation(self) -> np.ndarray:
        """Rotation matching the demo's synthetic palm angles."""
        yaw = 0.30 * math.sin(self.t * 0.42)
        pitch = 0.20 * math.cos(self.t * 0.31)
        roll = 0.16 * math.sin(self.t * 0.27)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
        ry = np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
        rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return rz @ ry @ rx
