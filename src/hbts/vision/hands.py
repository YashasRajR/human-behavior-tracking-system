"""Hand tracking and static gesture recognition.

Uses the MediaPipe ``GestureRecognizer`` bundle, which recognises seven
gestures out of the box: ``Closed_Fist``, ``Open_Palm``, ``Pointing_Up``,
``Thumb_Down``, ``Thumb_Up``, ``Victory`` and ``ILoveYou`` (``None`` otherwise).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from hbts.core.types import HandObservation


def _bbox(points: np.ndarray, width: int, height: int, pad: int = 12):
    x0, y0 = points[:, 0].min() - pad, points[:, 1].min() - pad
    x1, y1 = points[:, 0].max() + pad, points[:, 1].max() + pad
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), width - 1), min(int(y1), height - 1)
    return x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)


class HandAnalyzer:
    def __init__(self, model_path: Path, num_hands: int = 2, mirrored: bool = True) -> None:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        options = vision.GestureRecognizerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        # MediaPipe reports handedness assuming a mirrored (selfie) image.
        self._swap_handedness = not mirrored

    def process(self, mp_image, timestamp_ms: int, width: int, height: int) -> List[HandObservation]:
        result = self._recognizer.recognize_for_video(mp_image, timestamp_ms)
        hands: List[HandObservation] = []
        for i, pts in enumerate(result.hand_landmarks):
            lm = np.array([[p.x * width, p.y * height, p.z * width] for p in pts], dtype=np.float32)

            handedness = "Unknown"
            if i < len(result.handedness) and result.handedness[i]:
                handedness = result.handedness[i][0].category_name
                if self._swap_handedness:
                    handedness = {"Left": "Right", "Right": "Left"}.get(handedness, handedness)

            gesture, score = "None", 0.0
            if i < len(result.gestures) and result.gestures[i]:
                top = result.gestures[i][0]
                gesture, score = top.category_name or "None", float(top.score)

            hands.append(
                HandObservation(
                    handedness=handedness,
                    gesture=gesture,
                    gesture_score=score,
                    landmarks=lm,
                    bbox=_bbox(lm, width, height),
                )
            )
        return hands

    def close(self) -> None:
        self._recognizer.close()
