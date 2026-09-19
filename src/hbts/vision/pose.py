"""Upper-body posture analysis with the MediaPipe ``PoseLandmarker``.

A desk-facing webcam sees the user from the front, so posture is described by
quantities that are observable in that view:

* ``shoulder_tilt`` - roll of the shoulder line (leaning to one side)
* ``neck_angle``    - lateral inclination of the ear-midpoint over the
  shoulder-midpoint
* ``head_drop``     - vertical nose-to-shoulder distance normalised by shoulder
  width. Slouching or leaning towards the screen pushes the head down relative
  to the shoulders and shrinks this ratio; it is compared against the value
  captured during calibration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from hbts.core import geometry as geo
from hbts.core import landmarks as L
from hbts.core.types import PoseObservation


def extract_pose_features(lm: np.ndarray, visibility: np.ndarray, min_visibility: float = 0.5) -> Optional[PoseObservation]:
    """Posture descriptors from a (33, 3) pixel landmark array."""
    needed = (L.POSE_NOSE, L.POSE_LEFT_SHOULDER, L.POSE_RIGHT_SHOULDER)
    if any(visibility[i] < min_visibility for i in needed):
        return None

    ls, rs = lm[L.POSE_LEFT_SHOULDER, :2], lm[L.POSE_RIGHT_SHOULDER, :2]
    shoulder_mid = 0.5 * (ls + rs)
    shoulder_width = geo.distance(ls, rs)
    if shoulder_width < 1.0:
        return None

    le, re = lm[L.POSE_LEFT_EAR, :2], lm[L.POSE_RIGHT_EAR, :2]
    ears_visible = visibility[L.POSE_LEFT_EAR] >= min_visibility and visibility[L.POSE_RIGHT_EAR] >= min_visibility
    head_ref = 0.5 * (le + re) if ears_visible else lm[L.POSE_NOSE, :2]

    return PoseObservation(
        landmarks=lm,
        visibility=visibility,
        shoulder_tilt=geo.line_tilt(ls, rs),
        neck_angle=geo.angle_from_vertical(head_ref, shoulder_mid),
        head_drop=float(shoulder_mid[1] - lm[L.POSE_NOSE, 1]) / shoulder_width,
    )


class PoseAnalyzer:
    def __init__(self, model_path: Path) -> None:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def process(self, mp_image, timestamp_ms: int, width: int, height: int) -> Optional[PoseObservation]:
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            return None
        pts = result.pose_landmarks[0]
        lm = np.array([[p.x * width, p.y * height, p.z * width] for p in pts], dtype=np.float32)
        vis = np.array([getattr(p, "visibility", 1.0) or 0.0 for p in pts], dtype=np.float32)
        return extract_pose_features(lm, vis)

    def close(self) -> None:
        self._landmarker.close()
