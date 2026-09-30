"""Discrete gesture recognition on top of the continuous joint signals.

The continuous mirror is the main feature; this layer only adds a stable label
("FIST", "POINT", ...) for the HUD and for canned robot actions. A candidate
must win for `hold_frames` consecutive frames before it is accepted, which
kills the flicker that raw thresholding produces at gesture boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .robot_hand import FINGERS

# name -> friendly label shown in the HUD
LABELS = {
    "none": "READY",
    "open": "OPEN PALM",
    "fist": "FIST / GRIP",
    "point": "POINT",
    "pinch": "PINCH",
    "peace": "PEACE",
    "three": "THREE",
    "yoke": "YOKE",
    "wave": "WAVE",
}


@dataclass
class GestureReading:
    name: str = "none"
    confidence: float = 0.0
    stable_for: int = 0

    @property
    def label(self) -> str:
        return LABELS.get(self.name, self.name.upper())


def _c(signal: dict, name: str) -> float:
    return float(np.clip(signal.get(f"{name}_flex", 0.0), 0.0, 1.25))


def classify(signal: dict) -> tuple[str, float]:
    """Return (gesture, confidence) for one frame. Unambiguous only."""
    if not signal:
        return "none", 0.0

    th, ix, mi, ri, pi = (_c(signal, f) for f in FINGERS)
    four = np.array([ix, mi, ri, pi])
    mean4 = float(four.mean())
    mean3 = float(np.mean([mi, ri, pi]))
    # A pinch closes the thumb+index while the other three stay extended, so
    # mean3 must sit clearly *above* the pinched pair. A fist closes everything
    # (no contrast) and "three" leaves the index extended - both are rejected
    # here and fall through to their own rules.
    contrast = mean3 - 0.5 * (th + ix)

    if 0.30 <= ix <= 0.64 and th > 0.30 and mean3 > 0.62 and contrast > 0.18:
        return "pinch", float(np.clip(contrast, 0.0, 1.0))

    if ix < 0.52 and mean4 < 0.42 and th > 0.28:
        return "fist", float(np.clip(1.0 - mean4, 0.0, 1.0))

    if mean4 > 0.68 and th < 0.62:
        return "open", float(np.clip(mean4, 0.0, 1.0))

    if ix > 0.66 and mi < 0.42 and ri < 0.42 and pi < 0.48:
        return "point", float(np.clip(ix, 0.0, 1.0))

    if ix > 0.62 and mi > 0.62 and ri < 0.46 and pi < 0.52:
        return "peace", float(np.clip(0.5 * (ix + mi), 0.0, 1.0))

    if ix > 0.58 and mi > 0.58 and ri > 0.58 and pi < 0.50:
        return "three", float(np.clip((ix + mi + ri) / 3.0, 0.0, 1.0))

    if ix > 0.55 and pi > 0.55 and mi < 0.5 and ri < 0.5:
        return "yoke", float(np.clip(0.5 * (ix + pi), 0.0, 1.0))

    return "none", 0.0


class GestureRecognizer:
    """Adds temporal hysteresis to :func:`classify`."""

    def __init__(self, hold_frames: int = 4) -> None:
        self.hold_frames = int(hold_frames)
        self.reading = GestureReading()

    def reset(self) -> None:
        self.reading = GestureReading()

    def update(self, signal: dict, present: bool) -> GestureReading:
        if not present:
            self.reading = GestureReading()
            return self.reading

        name, conf = classify(signal)
        if name == self.reading.name:
            self.reading.stable_for += 1
            self.reading.confidence = 0.5 * self.reading.confidence + 0.5 * conf
        else:
            self.reading = GestureReading(name=name, confidence=conf, stable_for=1)

        if not self.reading.stable_for:
            return self.reading
        if self.reading.stable_for < self.hold_frames:
            # Not committed yet: report "none" so the HUD does not flicker.
            return GestureReading("none", self.reading.confidence, self.reading.stable_for)
        return self.reading


#: Canned robot actions, keyed by the number/letter that triggers them.
PRESETS: dict[str, dict] = {
    "1": {f: 0.03 for f in FINGERS},
    "2": {f: 0.50 for f in FINGERS},
    "3": {f: 1.00 for f in FINGERS},
    "4": {"thumb": 0.95, "index": 0.06, "middle": 0.06, "ring": 0.95, "pinky": 0.95},
    "5": {"thumb": 0.70, "index": 0.55, "middle": 0.20, "ring": 0.18, "pinky": 0.18},
    "6": {"thumb": 0.05, "index": 0.05, "middle": 0.05, "ring": 0.05, "pinky": 0.05},
    "7": {"thumb": 0.85, "index": 0.05, "middle": 0.05, "ring": 1.0, "pinky": 1.0},
}

PRESET_NAMES = {
    "1": "OPEN", "2": "HALF", "3": "FIST",
    "4": "YOKE", "5": "PINCH", "6": "FLAT RELAX", "7": "SCISSORS",
}
