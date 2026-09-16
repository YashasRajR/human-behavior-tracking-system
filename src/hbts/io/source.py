"""Frame sources with consistent timestamps.

Two behaviours are needed:

* **Live camera** - a background thread keeps only the newest frame so that
  slow inference never builds up latency (stale frames are dropped).
* **Video file** - every frame is processed in order and timestamps come from
  the file's frame rate, so analytics such as blinks-per-minute are correct
  even when processing runs slower or faster than real time.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    t: float          # seconds since the source was opened
    index: int


class VideoSource:
    def __init__(
        self,
        source: Union[int, str] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        mirror: bool = True,
    ) -> None:
        self.source = source
        self.mirror = mirror
        self.is_live = isinstance(source, int)
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            kind = "camera" if self.is_live else "video file"
            raise RuntimeError(f"Unable to open {kind}: {source}")

        if self.is_live:
            if width:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        self._index = 0
        self._t0 = time.perf_counter()
        self._latest: Optional[np.ndarray] = None
        self._latest_id = 0
        self._consumed_id = 0
        self._lock = threading.Lock()
        self._running = True
        self._thread: Optional[threading.Thread] = None
        if self.is_live:
            self._thread = threading.Thread(target=self._pump, name="camera-reader", daemon=True)
            self._thread.start()

    # -- live capture ---------------------------------------------------------
    def _pump(self) -> None:
        while self._running:
            ok, image = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest = image
                self._latest_id += 1

    def read(self, timeout: float = 2.0) -> Optional[Frame]:
        """Return the next frame, or ``None`` at end of stream / timeout."""
        if self.is_live:
            deadline = time.perf_counter() + timeout
            while time.perf_counter() < deadline:
                with self._lock:
                    if self._latest is not None and self._latest_id != self._consumed_id:
                        image = self._latest
                        self._consumed_id = self._latest_id
                        break
                time.sleep(0.002)
            else:
                return None
            t = time.perf_counter() - self._t0
        else:
            ok, image = self._cap.read()
            if not ok:
                return None
            t = self._index / self.fps

        if self.mirror:
            image = cv2.flip(image, 1)
        frame = Frame(image=image, t=t, index=self._index)
        self._index += 1
        return frame

    @property
    def progress(self) -> Optional[float]:
        if self.is_live or self.frame_count <= 0:
            return None
        return min(self._index / self.frame_count, 1.0)

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._cap.release()

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc) -> None:
        self.release()
