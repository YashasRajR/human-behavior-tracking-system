"""Face, eye, gaze and head-pose analysis.

``extract_face_features`` is a pure function of the landmark array, so it can
be tested against recorded landmarks without running a model.
``FaceAnalyzer`` wraps the MediaPipe ``FaceLandmarker`` and adds temporal
smoothing of the pose and gaze signals.

Left/right naming refers to the *image* side. With the default mirrored
(selfie) view this matches the user's own left and right.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from hbts.core import geometry as geo
from hbts.core import landmarks as L
from hbts.core.filters import OneEuroFilter
from hbts.core.types import FaceObservation


def _ordered_corners(lm: np.ndarray, a: int, b: int):
    """Return the two corner points ordered image-left to image-right."""
    pa, pb = lm[a], lm[b]
    return (pa, pb) if pa[0] <= pb[0] else (pb, pa)


def _eye_gaze(lm: np.ndarray, corners, lids, iris) -> tuple:
    left, right = _ordered_corners(lm, *corners)
    return geo.iris_ratio(lm[iris[0]], left, right, lm[lids[0]], lm[lids[1]])


def extract_face_features(lm: np.ndarray, width: int, height: int) -> FaceObservation:
    """Compute all per-frame face measurements from a (478, 3) pixel landmark array."""
    if lm.shape[0] < 478:
        raise ValueError("the refined 478-point face mesh (with iris) is required")

    oval = lm[list(L.FACE_OVAL), :2]
    x0, y0 = np.floor(oval.min(axis=0)).astype(int)
    x1, y1 = np.ceil(oval.max(axis=0)).astype(int)
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), width - 1), min(int(y1), height - 1)
    bbox = (x0, y0, max(x1 - x0, 1), max(y1 - y0, 1))

    # The "RIGHT_EYE" landmark set always appears on the image-left side.
    ear_img_left = geo.eye_aspect_ratio(lm[list(L.RIGHT_EYE_EAR), :2])
    ear_img_right = geo.eye_aspect_ratio(lm[list(L.LEFT_EYE_EAR), :2])

    mar = geo.mouth_aspect_ratio(
        lm[list(L.MOUTH_CORNERS), :2],
        lm[np.array(L.MOUTH_VERTICAL_PAIRS)][..., :2],
    )

    g1 = _eye_gaze(lm, L.RIGHT_EYE_CORNERS, L.RIGHT_EYE_LIDS, L.RIGHT_IRIS)
    g2 = _eye_gaze(lm, L.LEFT_EYE_CORNERS, L.LEFT_EYE_LIDS, L.LEFT_IRIS)
    gaze_h = 0.5 * (g1[0] + g2[0])
    gaze_v = 0.5 * (g1[1] + g2[1])

    # Order the paired anchors by image x so the correspondence with the 3-D
    # model never depends on how the mesh indices map to image sides.
    nose, chin, eye_a, eye_b, mouth_a, mouth_b = L.HEAD_POSE_ANCHORS
    eye_l, eye_r = _ordered_corners(lm, eye_a, eye_b)
    mouth_l, mouth_r = _ordered_corners(lm, mouth_a, mouth_b)
    anchors = np.stack([lm[nose], lm[chin], eye_l, eye_r, mouth_l, mouth_r])[:, :2]
    pose = geo.estimate_head_pose(anchors, width, height)
    pitch, yaw, roll = pose if pose is not None else (0.0, 0.0, 0.0)

    return FaceObservation(
        landmarks=lm,
        bbox=bbox,
        ear_left=ear_img_left,
        ear_right=ear_img_right,
        mar=mar,
        gaze_h=gaze_h,
        gaze_v=gaze_v,
        pitch=pitch,
        yaw=yaw,
        roll=roll,
    )


class FaceAnalyzer:
    """MediaPipe FaceLandmarker plus feature extraction and smoothing."""

    SMOOTHED = ("pitch", "yaw", "roll", "gaze_h", "gaze_v")

    def __init__(self, model_path: Path, min_cutoff: float = 1.2, beta: float = 0.03) -> None:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._filters: Dict[str, OneEuroFilter] = {
            name: OneEuroFilter(min_cutoff=min_cutoff, beta=beta) for name in self.SMOOTHED
        }

    def process(self, mp_image, timestamp_ms: int, t: float, width: int, height: int) -> Optional[FaceObservation]:
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            for f in self._filters.values():
                f.reset()
            return None

        pts = result.face_landmarks[0]
        lm = np.array([[p.x * width, p.y * height, p.z * width] for p in pts], dtype=np.float32)
        obs = extract_face_features(lm, width, height)
        for name, flt in self._filters.items():
            setattr(obs, name, flt(getattr(obs, name), t))
        return obs

    def close(self) -> None:
        self._landmarker.close()
