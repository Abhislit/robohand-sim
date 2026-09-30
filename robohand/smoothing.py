"""Signal smoothing for real-time control loops.

Uses the One Euro filter (Casiez et al., CHI 2012): heavy smoothing when the
signal is still (kills jitter) and light smoothing when it moves fast (keeps
latency low). A plain low-pass or EMA cannot do both.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable

import numpy as np


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff_hz, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroFilter:
    """One Euro filter. Works on scalars or numpy arrays."""

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.035,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = None
        self._dx = None
        self._t = None

    def __call__(self, value, t: float):
        x = np.asarray(value, dtype=float)
        if self._x is None:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t
            return self._x.copy()

        dt = t - (self._t if self._t is not None else t)
        self._t = t
        if dt <= 1e-6:
            return self._x.copy()

        dx = (x - self._x) / dt
        ad = _alpha(self.d_cutoff, dt)
        self._dx = ad * dx + (1.0 - ad) * self._dx

        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = _alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x.copy()

    @property
    def value(self):
        return self._x.copy() if self._x is not None else None


class PoseSmoother:
    """Smoots a whole pose (dict of named signals) with per-channel filters.

    `aggressiveness` in [0, 1] trades latency for stability.
    """

    def __init__(
        self,
        channels: Iterable[str],
        min_cutoff: float = 1.6,
        beta: float = 0.05,
        aggressiveness: float = 0.5,
    ) -> None:
        self.channels = list(channels)
        self.base_min_cutoff = float(min_cutoff)
        self.base_beta = float(beta)
        self.aggressiveness = float(np.clip(aggressiveness, 0.0, 1.0))
        self._filters = {c: OneEuroFilter(*self._params()) for c in self.channels}
        self._last_seen: float | None = None

    def _params(self) -> tuple[float, float, float]:
        a = self.aggressiveness
        min_cutoff = self.base_min_cutoff * (1.0 + 3.0 * a)
        beta = self.base_beta * (1.0 + 8.0 * a)
        return min_cutoff, beta, 1.0

    def set_aggressiveness(self, value: float) -> None:
        value = float(np.clip(value, 0.0, 1.0))
        if abs(value - self.aggressiveness) < 1e-3:
            return
        self.aggressiveness = value
        # Re-derive existing filters so tuning takes effect immediately.
        for key, filt in self._filters.items():
            old = filt.value
            filt = OneEuroFilter(*self._params())
            if old is not None:
                filt._x = np.asarray(old, dtype=float)
                filt._dx = np.zeros_like(filt._x)
            self._filters[key] = filt

    def reset(self) -> None:
        for f in self._filters.values():
            f.reset()
        self._last_seen = None

    def __call__(self, pose: dict, t: float, seen: bool = True) -> dict:
        if not seen:
            # Hand lost: hold last pose, but stop integrating time so the
            # filters do not jump when the hand reappears.
            self._last_seen = t
            return {k: (f.value if f.value is not None else pose.get(k, 0.0))
                    for k, f in self._filters.items()}

        out = {}
        for key in self.channels:
            raw = pose.get(key, 0.0)
            out[key] = float(self._filters[key](raw, t))
        self._last_seen = t
        return out


class FpsMeter:
    """Rolling frames-per-second estimate."""

    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self, t: float) -> float:
        self._times.append(t)
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        if span <= 1e-6:
            return 0.0
        return (len(self._times) - 1) / span
