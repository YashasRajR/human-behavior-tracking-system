"""Fatigue indicators: PERCLOS, microsleeps and yawns.

PERCLOS (PERcentage of eyelid CLOSure over time) is the drowsiness measure
most strongly correlated with lapses in vigilance in driving studies. Here it
is computed as the fraction of samples in a sliding window during which the
eyes were classified as closed.

References:
* W. W. Wierwille et al., "Research on vehicle-based driver status/performance
  monitoring", NHTSA report, 1994.
* D. F. Dinges and R. Grace, "PERCLOS: A valid psychophysiological measure of
  alertness as assessed by psychomotor vigilance", FHWA-MCRT-98-006, 1998.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from hbts.core.filters import TimeWindow


class PerclosMeter:
    def __init__(self, window: float = 60.0, warmup: float = 15.0) -> None:
        self._window = TimeWindow(window)
        self.warmup = warmup

    def update(self, closed: bool, t: float) -> float:
        self._window.push(t, 1.0 if closed else 0.0)
        return self.value

    @property
    def value(self) -> float:
        return self._window.mean(0.0)

    @property
    def ready(self) -> bool:
        """True once enough history exists for PERCLOS to be meaningful."""
        return self._window.span() >= self.warmup

    def reset(self) -> None:
        self._window.clear()


@dataclass
class YawnUpdate:
    yawning: bool
    started: bool   # a new yawn was confirmed on this frame


class YawnDetector:
    """A yawn is a wide mouth opening sustained for ``min_seconds``.

    Speaking produces short MAR peaks; requiring a sustained opening separates
    the two without a learned model.
    """

    def __init__(self, mar_threshold: float = 0.55, min_seconds: float = 1.2, window: float = 300.0) -> None:
        self.mar_threshold = mar_threshold
        self.min_seconds = min_seconds
        self.window = window
        self.total = 0
        self._open_since: Optional[float] = None
        self._counted = False
        self._times: Deque[float] = deque()

    def update(self, mar: float, t: float) -> YawnUpdate:
        started = False
        if mar >= self.mar_threshold:
            if self._open_since is None:
                self._open_since = t
                self._counted = False
            elif not self._counted and t - self._open_since >= self.min_seconds:
                self._counted = True
                started = True
                self.total += 1
                self._times.append(t)
        else:
            self._open_since = None
            self._counted = False

        while self._times and self._times[0] < t - self.window:
            self._times.popleft()
        return YawnUpdate(yawning=self._counted, started=started)

    def recent(self) -> int:
        """Yawns within the trailing window (default five minutes)."""
        return len(self._times)


def fatigue_level(perclos: float, perclos_threshold: float, recent_yawns: int, microsleep: bool) -> str:
    """Map indicators to a three-level label: ``Alert``, ``Tired`` or ``Drowsy``."""
    if microsleep or perclos >= 2.0 * perclos_threshold:
        return "Drowsy"
    if perclos >= perclos_threshold or recent_yawns >= 3:
        return "Tired"
    return "Alert"
