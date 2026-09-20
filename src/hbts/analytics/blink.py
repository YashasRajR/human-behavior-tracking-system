"""Blink detection with adaptive thresholds and hysteresis.

The first prototype compared the EAR of one eye against a fixed 0.15
threshold. That fails for people whose resting EAR is naturally low or high,
double-counts blinks when the EAR hovers around the threshold, and counts long
eye closures as blinks.

This detector:

* uses both eyes (mean EAR) against a *personal* open-eye baseline obtained
  during calibration,
* applies two thresholds (close below ``close_ratio``, reopen above
  ``open_ratio``) so noise near a single threshold cannot toggle the state,
* only counts closures lasting between ``min_duration`` and ``max_duration``
  as blinks (typical human blinks last 100-400 ms),
* reports blink rate over a sliding window instead of a whole-session average.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class BlinkUpdate:
    closed: bool
    blinked: bool                  # a blink completed on this frame
    closure_seconds: float         # duration of the current closure (0 if open)


class BlinkDetector:
    def __init__(
        self,
        close_ratio: float = 0.72,
        open_ratio: float = 0.82,
        min_duration: float = 0.05,
        max_duration: float = 0.50,
        rate_window: float = 60.0,
    ) -> None:
        if not 0 < close_ratio < open_ratio <= 1.5:
            raise ValueError("expected 0 < close_ratio < open_ratio")
        self.close_ratio = close_ratio
        self.open_ratio = open_ratio
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.rate_window = rate_window

        self.total = 0
        self.long_closures = 0
        self._closed = False
        self._closed_since: Optional[float] = None
        self._blink_times: Deque[float] = deque()
        self._t_first: Optional[float] = None
        self._t_last: float = 0.0

    @property
    def closed(self) -> bool:
        return self._closed

    def update(self, ear: float, t: float, baseline: float) -> BlinkUpdate:
        if self._t_first is None:
            self._t_first = t
        self._t_last = t

        blinked = False
        if not self._closed:
            if ear < self.close_ratio * baseline:
                self._closed = True
                self._closed_since = t
        elif ear > self.open_ratio * baseline:
            duration = t - (self._closed_since if self._closed_since is not None else t)
            self._closed = False
            self._closed_since = None
            if self.min_duration <= duration <= self.max_duration:
                blinked = True
                self.total += 1
                self._blink_times.append(t)
            elif duration > self.max_duration:
                self.long_closures += 1

        self._evict(t)
        closure = t - self._closed_since if (self._closed and self._closed_since is not None) else 0.0
        return BlinkUpdate(closed=self._closed, blinked=blinked, closure_seconds=closure)

    def mark_absent(self) -> None:
        """Forget an in-progress closure when the face is lost."""
        self._closed = False
        self._closed_since = None

    def _evict(self, now: float) -> None:
        cutoff = now - self.rate_window
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()

    def rate_per_minute(self) -> float:
        """Blinks per minute over the sliding window.

        Early in a session the window is only partially filled, so the count is
        scaled by the observed duration (with a 10-second floor to avoid
        wildly inflated estimates from the first blink).
        """
        if self._t_first is None:
            return 0.0
        observed = min(self._t_last - self._t_first, self.rate_window)
        observed = max(observed, 10.0)
        return len(self._blink_times) * 60.0 / observed
