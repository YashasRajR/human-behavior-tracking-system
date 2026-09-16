"""Shared data structures passed between the perception, analytics and UI layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]  # x, y, width, height


@dataclass
class FaceObservation:
    """Per-frame measurements for a single face.

    Coordinates are in pixels. ``*_left`` / ``*_right`` refer to the eye on the
    image-left / image-right side, which in the default mirrored view is the
    user's own left / right eye.
    """

    landmarks: np.ndarray                 # (478, 3) pixel coordinates, z in pixel units
    bbox: BBox
    ear_left: float
    ear_right: float
    mar: float
    gaze_h: float                         # 0 = image-left, 1 = image-right
    gaze_v: float                         # 0 = up, 1 = down
    pitch: float
    yaw: float
    roll: float

    @property
    def ear(self) -> float:
        return 0.5 * (self.ear_left + self.ear_right)

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2


@dataclass
class HandObservation:
    """A detected hand and its recognised gesture."""

    handedness: str                        # "Left" / "Right", from the user's perspective
    gesture: str                           # MediaPipe category name, e.g. "Thumb_Up"
    gesture_score: float
    landmarks: np.ndarray                  # (21, 3) pixel coordinates
    bbox: BBox

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2


@dataclass
class PoseObservation:
    """Upper-body posture descriptors derived from BlazePose landmarks."""

    landmarks: np.ndarray                  # (33, 3) pixel coordinates
    visibility: np.ndarray                 # (33,)
    shoulder_tilt: float                   # degrees, 0 = level
    neck_angle: float                      # degrees between ear-shoulder line and vertical
    head_drop: float                       # nose height above shoulders / shoulder width


@dataclass
class FrameAnalysis:
    """Everything the perception layer extracted from one frame."""

    t: float
    index: int
    width: int
    height: int
    face: Optional[FaceObservation] = None
    hands: List[HandObservation] = field(default_factory=list)
    pose: Optional[PoseObservation] = None
    inference_ms: float = 0.0
