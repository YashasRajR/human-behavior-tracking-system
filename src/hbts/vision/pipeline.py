"""Runs the enabled analyzers on a frame and returns a single ``FrameAnalysis``."""

from __future__ import annotations

import time
from typing import Any, Dict

import cv2
import numpy as np

from hbts.core.types import FrameAnalysis
from hbts.io.assets import ensure_model


class PerceptionPipeline:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        import mediapipe as mp

        self._mp = mp
        modules = cfg["modules"]
        models = cfg["models"]
        smoothing = cfg["smoothing"]
        mirrored = bool(cfg["source"].get("mirror", True))

        self.face = self.hands = self.pose = None
        if modules.get("face", True):
            from hbts.vision.face import FaceAnalyzer

            self.face = FaceAnalyzer(
                ensure_model(models["face"], models["dir"]),
                min_cutoff=smoothing["min_cutoff"],
                beta=smoothing["beta"],
            )
        if modules.get("hands", True):
            from hbts.vision.hands import HandAnalyzer

            self.hands = HandAnalyzer(ensure_model(models["hands"], models["dir"]), mirrored=mirrored)
        if modules.get("pose", True):
            from hbts.vision.pose import PoseAnalyzer

            self.pose = PoseAnalyzer(ensure_model(models["pose"], models["dir"]))

        self._last_ts = -1

    def _timestamp_ms(self, t: float) -> int:
        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        ts = max(int(t * 1000.0), self._last_ts + 1)
        self._last_ts = ts
        return ts

    def analyze(self, image: np.ndarray, t: float, index: int) -> FrameAnalysis:
        h, w = image.shape[:2]
        started = time.perf_counter()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = self._timestamp_ms(t)

        analysis = FrameAnalysis(t=t, index=index, width=w, height=h)
        if self.face is not None:
            analysis.face = self.face.process(mp_image, ts, t, w, h)
        if self.hands is not None:
            analysis.hands = self.hands.process(mp_image, ts, w, h)
        if self.pose is not None:
            analysis.pose = self.pose.process(mp_image, ts, w, h)
        analysis.inference_ms = (time.perf_counter() - started) * 1000.0
        return analysis

    def close(self) -> None:
        for analyzer in (self.face, self.hands, self.pose):
            if analyzer is not None:
                analyzer.close()
