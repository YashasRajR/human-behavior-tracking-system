"""Session recording.

The prototype rewrote a JSON file on every frame and re-opened an Excel
workbook every tenth frame, which blocks the video loop on disk I/O and keeps
only the latest sample. The recorder here:

* writes append-only JSON Lines (one object per line; streamable, robust to
  crashes, trivially loaded with ``pandas.read_json(lines=True)``) plus a CSV
  mirror for spreadsheet users,
* samples metrics at a fixed rate (default 10 Hz) independent of camera FPS,
* performs all file I/O on a background thread fed by a queue.

Session layout::

    sessions/20260930-142501/
        config.json      resolved configuration used for the run
        metrics.jsonl    time series of BehaviorState.metrics()
        metrics.csv      same data, flat CSV
        events.jsonl     completed events (debounced and instant)
        summary.json     aggregate statistics written on close
        snapshots/       frames saved on demand
"""

from __future__ import annotations

import csv
import json
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

CSV_FIELDS = [
    "t", "face", "calibrating", "attention", "on_screen", "ear", "eyes_closed", "blink_count", "blink_rate",
    "perclos", "perclos_ready", "mar", "yawn_count", "fatigue", "yaw", "pitch", "roll", "gaze_h", "gaze_v",
    "head", "gaze", "posture", "gestures",
]

_STOP = object()


class SessionRecorder:
    def __init__(self, root: str = "sessions", sample_hz: float = 10.0,
                 config: Optional[Dict[str, Any]] = None, name: Optional[str] = None) -> None:
        stamp = name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = Path(root) / stamp
        suffix = 1
        while self.path.exists():
            self.path = Path(root) / f"{stamp}-{suffix}"
            suffix += 1
        (self.path / "snapshots").mkdir(parents=True, exist_ok=True)

        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.interval = 1.0 / sample_hz if sample_hz > 0 else 0.0
        self.paused = False
        self.samples = 0
        self._next_sample_t: Optional[float] = None
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=10000)
        self._closed = False

        if config is not None:
            (self.path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

        self._metrics = (self.path / "metrics.jsonl").open("w", encoding="utf-8")
        self._events = (self.path / "events.jsonl").open("w", encoding="utf-8")
        self._csv_file = (self.path / "metrics.csv").open("w", encoding="utf-8", newline="")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        self._csv.writeheader()

        self._thread = threading.Thread(target=self._worker, name="session-writer", daemon=True)
        self._thread.start()

    # -- producer side (video thread) ----------------------------------------
    def record(self, metrics: Dict[str, Any]) -> bool:
        """Queue a metrics sample if the sampling interval has elapsed."""
        if self._closed or self.paused:
            return False
        t = float(metrics.get("t", 0.0))
        if self._next_sample_t is not None and t < self._next_sample_t:
            return False
        # Schedule against a fixed grid so the average rate matches sample_hz
        # even when frames arrive at a rate that is not a multiple of it.
        base = t if self._next_sample_t is None else self._next_sample_t
        self._next_sample_t = max(base + self.interval, t)
        self.samples += 1
        self._put(("metrics", dict(metrics)))
        return True

    def event(self, event_dict: Dict[str, Any]) -> None:
        if not self._closed:
            self._put(("event", dict(event_dict)))

    def snapshot_path(self, t: float) -> Path:
        return self.path / "snapshots" / f"snapshot_{t:09.2f}s.png"

    def _put(self, item: Any) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:  # never block the video loop
            pass

    # -- consumer side (writer thread) ---------------------------------------
    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                break
            kind, payload = item
            if kind == "metrics":
                self._metrics.write(json.dumps(payload) + "\n")
                row = dict(payload)
                row["gestures"] = "|".join(payload.get("gestures", []))
                self._csv.writerow(row)
            elif kind == "event":
                self._events.write(json.dumps(payload) + "\n")
        for handle in (self._metrics, self._events, self._csv_file):
            handle.flush()

    def close(self, summary: Optional[Dict[str, Any]] = None) -> Path:
        if self._closed:
            return self.path
        self._closed = True
        self._queue.put(_STOP)
        self._thread.join(timeout=5.0)
        for handle in (self._metrics, self._events, self._csv_file):
            handle.close()
        data = {"started_at": self.started_at,
                "ended_at": datetime.now().isoformat(timespec="seconds"),
                "samples": self.samples}
        data.update(summary or {})
        (self.path / "summary.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        return self.path
