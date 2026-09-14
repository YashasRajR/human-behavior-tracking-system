"""Signal conditioning for noisy per-frame measurements.

Landmark-derived signals (angles, ratios, iris positions) jitter by a few
pixels every frame. Plain moving averages remove the jitter but add lag that
makes the dashboard feel sluggish. The One Euro filter adapts its cut-off
frequency to the signal speed: heavy smoothing when the signal is still, light
smoothing when it moves quickly.

Reference: G. Casiez, N. Roussel and D. Vogel, "1 Euro Filter: A Simple
Speed-based Low-pass Filter for Noisy Input in Interactive Systems", CHI 2012.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Tuple


def _smoothing_factor(dt: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Adaptive low-pass filter for a scalar signal sampled at irregular times."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cut-off frequencies must be positive")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: Optional[float] = None
        self._dx: float = 0.0
        self._t: Optional[float] = None

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0
        self._t = None

    @property
    def value(self) -> Optional[float]:
        return self._x

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None:
            self._x, self._t = float(x), float(t)
            return self._x

        dt = t - self._t
        if dt <= 0:
            # Duplicate or out-of-order timestamp: keep the previous estimate.
            return self._x

        dx = (x - self._x) / dt
        a_d = _smoothing_factor(dt, self.d_cutoff)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx

        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _smoothing_factor(dt, cutoff)
        self._x = a * x + (1.0 - a) * self._x
        self._t = float(t)
        return self._x


class EMA:
    """Exponential moving average with a time constant expressed in seconds.

    Unlike a per-sample alpha, the time constant keeps the smoothing behaviour
    identical at 15 FPS and at 60 FPS.
    """

    def __init__(self, time_constant: float = 0.5) -> None:
        if time_constant <= 0:
            raise ValueError("time_constant must be positive")
        self.time_constant = time_constant
        self._value: Optional[float] = None
        self._t: Optional[float] = None

    def reset(self) -> None:
        self._value = None
        self._t = None

    @property
    def value(self) -> Optional[float]:
        return self._value

    def __call__(self, x: float, t: float) -> float:
        if self._value is None or self._t is None:
            self._value, self._t = float(x), float(t)
            return self._value
        dt = max(t - self._t, 0.0)
        alpha = 1.0 - math.exp(-dt / self.time_constant)
        self._value += alpha * (x - self._value)
        self._t = float(t)
        return self._value


class TimeWindow:
    """Fixed-duration sliding window of ``(t, value)`` samples."""

    def __init__(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("window length must be positive")
        self.seconds = seconds
        self._items: Deque[Tuple[float, float]] = deque()

    def push(self, t: float, value: float) -> None:
        self._items.append((t, float(value)))
        self._evict(t)

    def _evict(self, now: float) -> None:
        cutoff = now - self.seconds
        while self._items and self._items[0][0] < cutoff:
            self._items.popleft()

    def values(self) -> List[float]:
        return [v for _, v in self._items]

    def mean(self, default: float = 0.0) -> float:
        if not self._items:
            return default
        return sum(v for _, v in self._items) / len(self._items)

    def span(self) -> float:
        """Time covered by the samples currently in the window."""
        if len(self._items) < 2:
            return 0.0
        return self._items[-1][0] - self._items[0][0]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
