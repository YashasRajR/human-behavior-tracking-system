"""Per-user calibration.

Fixed thresholds do not transfer between people, cameras and seating
positions: resting EAR varies from roughly 0.22 to 0.35, the webcam is rarely
exactly in front of the face, and a "neutral" head pose depends on where the
screen is. During the first few seconds the user is asked to look at the
screen normally, and the engine records personal baselines. All downstream
analytics are then expressed relative to these baselines.

Robust statistics are used throughout: the open-eye EAR baseline is the 75th
percentile (blinks during calibration pull the mean down), and angles use the
median.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Baseline:
    ear: float = 0.28
    yaw: float = 0.0
    pitch: float = 0.0
    gaze_h: float = 0.5
    gaze_v: float = 0.45
    head_drop: Optional[float] = None
    neck_angle: float = 0.0
    calibrated: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class Calibrator:
    FIELDS = ("ear", "yaw", "pitch", "gaze_h", "gaze_v", "head_drop", "neck_angle")

    def __init__(self, seconds: float = 4.0, min_samples: int = 30, default_ear: float = 0.28) -> None:
        self.seconds = seconds
        self.min_samples = min_samples
        self.baseline = Baseline(ear=default_ear)
        self._samples: Dict[str, List[float]] = {k: [] for k in self.FIELDS}
        self._t_start: Optional[float] = None
        self._active = False

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self._samples = {k: [] for k in self.FIELDS}
        self._t_start = None
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def progress(self, t: float) -> float:
        if not self._active:
            return 1.0 if self.baseline.calibrated else 0.0
        if self._t_start is None:
            return 0.0
        by_time = (t - self._t_start) / self.seconds
        by_count = len(self._samples["ear"]) / float(self.min_samples)
        return float(np.clip(min(by_time, by_count), 0.0, 1.0))

    # -- sampling ------------------------------------------------------------
    def add(
        self,
        t: float,
        ear: Optional[float] = None,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        gaze_h: Optional[float] = None,
        gaze_v: Optional[float] = None,
        head_drop: Optional[float] = None,
        neck_angle: Optional[float] = None,
    ) -> bool:
        """Add one frame of measurements. Returns ``True`` when calibration completes."""
        if not self._active:
            return False
        if ear is None:
            # No face: calibration waits rather than capturing a bad baseline.
            return False
        if self._t_start is None:
            self._t_start = t

        values = dict(ear=ear, yaw=yaw, pitch=pitch, gaze_h=gaze_h, gaze_v=gaze_v,
                      head_drop=head_drop, neck_angle=neck_angle)
        for key, value in values.items():
            if value is not None:
                self._samples[key].append(float(value))

        if self.progress(t) >= 1.0:
            self._finish()
            return True
        return False

    def _finish(self) -> None:
        s = self._samples
        b = Baseline(calibrated=True)
        b.ear = float(np.percentile(s["ear"], 75))
        b.yaw = float(np.median(s["yaw"])) if s["yaw"] else 0.0
        b.pitch = float(np.median(s["pitch"])) if s["pitch"] else 0.0
        b.gaze_h = float(np.median(s["gaze_h"])) if s["gaze_h"] else 0.5
        b.gaze_v = float(np.median(s["gaze_v"])) if s["gaze_v"] else 0.45
        b.head_drop = float(np.median(s["head_drop"])) if s["head_drop"] else None
        b.neck_angle = float(np.median(s["neck_angle"])) if s["neck_angle"] else 0.0
        self.baseline = b
        self._active = False
