"""Application loop: source -> perception -> analytics -> dashboard / recorder."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2

from hbts.analytics.engine import BehaviorEngine, BehaviorState
from hbts.analytics.events import Event
from hbts.io.recorder import SessionRecorder
from hbts.io.report import generate_report
from hbts.io.source import VideoSource
from hbts.ui.dashboard import Dashboard, ViewStatus
from hbts.vision.pipeline import PerceptionPipeline

KEY_ESC = 27


class Application:
    def __init__(self, cfg: Dict[str, Any], video: Optional[str] = None, headless: bool = False,
                 save_video: Optional[str] = None, session_name: Optional[str] = None) -> None:
        self.cfg = cfg
        self.headless = headless
        src_cfg = cfg["source"]
        source: Union[int, str] = video if video else int(src_cfg["camera"])
        self.source = VideoSource(source, src_cfg.get("width"), src_cfg.get("height"),
                                  mirror=bool(src_cfg.get("mirror", True)) and video is None)
        self.pipeline = PerceptionPipeline(cfg)

        rec_cfg = cfg["recording"]
        self.recorder: Optional[SessionRecorder] = None
        if rec_cfg.get("enabled", True):
            self.recorder = SessionRecorder(rec_cfg["dir"], rec_cfg["sample_hz"], config=cfg, name=session_name)

        ui = cfg["ui"]
        self.dashboard = Dashboard(panel_width=int(ui["panel_width"]))
        self.status = ViewStatus(
            recording=self.recorder is not None,
            privacy=bool(ui.get("privacy", False)),
            show_mesh=bool(ui.get("show_mesh", True)),
            source_label=f"Camera {source}" if isinstance(source, int) else Path(str(source)).name,
            hold_seconds=float(cfg["commands"]["hold_seconds"]),
        )
        self.engine = BehaviorEngine(cfg, on_event=self._on_event)

        self._writer: Optional[cv2.VideoWriter] = None
        self._save_video = save_video
        self._last_canvas = None
        self._last_state = BehaviorState()
        self._last_image = None
        self._last_analysis = None
        self._fps = 0.0
        self._last_tick = time.perf_counter()
        self._last_print = 0.0

    # -- event plumbing -------------------------------------------------------
    def _on_event(self, event: Event) -> None:
        starting = event.end is None or event.end == event.start
        if starting:
            self.dashboard.notify(event.message or event.kind, event.severity, event.start)
        if event.end is not None and self.recorder is not None and self.status.recording:
            self.recorder.event(event.to_dict())

    def _run_command(self, command: str, t: float) -> None:
        if command == "toggle_privacy":
            self.status.privacy = not self.status.privacy
            self.dashboard.notify(f"Privacy mode {'on' if self.status.privacy else 'off'}", "info", t)
        elif command == "toggle_recording":
            if self.recorder is None:
                self.dashboard.notify("Recording is disabled in the configuration", "warning", t)
                return
            self.status.recording = not self.status.recording
            self.recorder.paused = not self.status.recording
            self.dashboard.notify(f"Recording {'resumed' if self.status.recording else 'paused'}", "info", t)
        elif command == "bookmark":
            self.engine.mark("bookmark", t, "Bookmark added")
        elif command == "snapshot":
            self._snapshot(t)

    def _snapshot(self, t: float) -> None:
        if self._last_canvas is None:
            return
        if self.recorder is not None:
            path = self.recorder.snapshot_path(t)
        else:
            path = Path("snapshots") / f"snapshot_{int(time.time())}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self._last_canvas)
        self.engine.mark("snapshot", t, "Snapshot saved", data={"file": path.name})

    # -- keyboard -------------------------------------------------------------
    def _handle_key(self, key: int, t: float) -> bool:
        """Returns False when the application should exit."""
        if key in (KEY_ESC, ord("q"), ord("Q")):
            return False
        mapping = {
            ord("h"): lambda: setattr(self.status, "show_help", not self.status.show_help),
            ord("m"): lambda: setattr(self.status, "show_mesh", not self.status.show_mesh),
            ord("p"): lambda: self._run_command("toggle_privacy", t),
            ord("r"): lambda: self._run_command("toggle_recording", t),
            ord("b"): lambda: self._run_command("bookmark", t),
            ord("s"): lambda: self._run_command("snapshot", t),
            ord("c"): self._recalibrate,
            ord(" "): lambda: setattr(self.status, "paused", not self.status.paused),
        }
        action = mapping.get(key) or mapping.get(ord(chr(key).lower()) if 0 <= key < 256 else -1)
        if action is not None:
            action()
        return True

    def _recalibrate(self) -> None:
        self.engine.recalibrate()
        self.dashboard.notify("Recalibrating", "info", self._last_state.t)

    # -- main loop ------------------------------------------------------------
    def _step(self) -> bool:
        """Read, analyse and render one frame. Returns False at end of stream."""
        frame = self.source.read()
        if frame is None:
            return False
        analysis = self.pipeline.analyze(frame.image, frame.t, frame.index)
        state = self.engine.update(analysis)
        self._last_state = state
        self._last_image, self._last_analysis = frame.image, analysis

        if state.command is not None and state.command.fired:
            self._run_command(state.command.fired, state.t)
        if self.recorder is not None and self.status.recording:
            self.recorder.record(state.metrics())

        now = time.perf_counter()
        dt, self._last_tick = now - self._last_tick, now
        if dt > 0:
            self._fps = 1.0 / dt if self._fps == 0 else 0.9 * self._fps + 0.1 / dt
        self.status.fps = self._fps
        self.status.inference_ms = analysis.inference_ms
        self.status.elapsed = state.t
        self.status.progress = self.source.progress

        self._last_canvas = self.dashboard.render(frame.image, analysis, state, self.status)
        self._write_video(self._last_canvas)
        return True

    def run(self) -> Optional[Path]:
        window = self.cfg["ui"]["window"]
        if not self.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            if self.cfg["ui"].get("fullscreen"):
                cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self._last_tick = time.perf_counter()
        try:
            while True:
                if self.headless:
                    if not self._step():
                        break
                    if self._last_state.t - self._last_print >= 5.0 and self.source.progress is not None:
                        self._last_print = self._last_state.t
                        print(f"\r  processed {self.source.progress * 100:5.1f}%", end="", flush=True)
                    continue

                if not self.status.paused or self._last_canvas is None:
                    if not self._step():
                        break
                    canvas = self._last_canvas
                else:
                    # Re-render the frozen frame so overlays (help, pause) stay interactive.
                    canvas = self.dashboard.render(self._last_image, self._last_analysis,
                                                   self._last_state, self.status)

                cv2.imshow(window, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key != 255 and not self._handle_key(key, self._last_state.t):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        except KeyboardInterrupt:
            pass
        return self.shutdown()

    def _write_video(self, canvas) -> None:
        if not self._save_video:
            return
        if self._writer is None:
            h, w = canvas.shape[:2]
            fps = self.source.fps if not self.source.is_live else 20.0
            self._writer = cv2.VideoWriter(self._save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        self._writer.write(canvas)

    def shutdown(self) -> Optional[Path]:
        t = self._last_state.t
        self.engine.events.close_all(t)
        self.source.release()
        self.pipeline.close()
        if self._writer is not None:
            self._writer.release()
        if not self.headless:
            cv2.destroyAllWindows()
        if self.headless and self.source.progress is not None:
            print(f"\r  processed {self.source.progress * 100:5.1f}%")
        if self.recorder is None:
            return None
        session = self.recorder.close(self.engine.summary(t))
        report = generate_report(str(session))
        print(f"Session saved to {session}")
        print(f"Report: {report}")
        return session
