"""Dashboard layout.

    +--------------------------------------------------------------+
    | HBTS   REC  PRIVACY                       00:12:41  29.8 FPS |  top bar
    +------------------------------------------+-------------------+
    |                                          | ATTENTION         |
    |                                          |  82  Focused      |
    |              video / privacy view        |  ~~~~~~~ 60 s     |
    |          (landmarks, command rings,      +-------------------+
    |           alerts, notifications)         | EYES & FATIGUE    |
    |                                          +-------------------+
    |                                          | GAZE & HEAD       |
    |                                          +-------------------+
    |                                          | POSTURE & HANDS   |
    +------------------------------------------+-------------------+

The video is always scaled to a fixed height so the side panel layout is
identical for every camera resolution.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from hbts.analytics.engine import BehaviorState
from hbts.core.types import FrameAnalysis
from hbts.ui import overlays as O
from hbts.ui import theme as T
from hbts.ui import widgets as W


@dataclass
class ViewStatus:
    recording: bool = True
    privacy: bool = False
    show_mesh: bool = True
    show_help: bool = False
    paused: bool = False
    fps: float = 0.0
    inference_ms: float = 0.0
    elapsed: float = 0.0
    source_label: str = "Camera 0"
    progress: Optional[float] = None
    hold_seconds: float = 1.2


@dataclass
class _History:
    t: Deque[float] = field(default_factory=lambda: deque(maxlen=900))
    attention: Deque[float] = field(default_factory=lambda: deque(maxlen=900))


def attention_label(score: float, face: bool, calibrating: bool = False) -> Tuple[str, Tuple[int, int, int]]:
    if calibrating:
        return "Calibrating", T.ACCENT
    if not face:
        return "Away", T.MUTED
    if score >= 75:
        return "Focused", T.GOOD
    if score >= 50:
        return "Wandering", T.WARNING
    return "Distracted", T.CRITICAL


def _fmt_clock(seconds: float) -> str:
    s = int(max(seconds, 0))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


class Dashboard:
    def __init__(self, panel_width: int = 380, video_height: int = T.VIDEO_HEIGHT,
                 history_seconds: float = 60.0) -> None:
        self.panel_width = panel_width
        self.video_height = video_height
        self.history_seconds = history_seconds
        self._history = _History()
        self._toasts: List[Tuple[str, str, float]] = []

    # -- notifications --------------------------------------------------------
    def notify(self, message: str, severity: str, t: float) -> None:
        self._toasts.append((message, severity, t))
        self._toasts = self._toasts[-8:]

    # -- main entry -----------------------------------------------------------
    def render(self, image: np.ndarray, frame: Optional[FrameAnalysis], state: BehaviorState,
               status: ViewStatus) -> np.ndarray:
        h0, w0 = image.shape[:2]
        scale = self.video_height / float(h0)
        vw = int(round(w0 * scale))
        vh = self.video_height

        if status.privacy:
            video = O.privacy_canvas((vh, vw, 3))
        else:
            video = cv2.resize(image, (vw, vh), interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)

        if frame is not None and (status.show_mesh or status.privacy):
            if frame.pose is not None:
                O.draw_pose(video, frame.pose, scale)
            if frame.face is not None:
                O.draw_face(video, frame.face, scale, strong=status.privacy)
            O.draw_hands(video, frame.hands, scale)

        self._push_history(state)
        self._video_layers(video, frame, state, status, scale)

        canvas = np.empty((T.TOP_BAR_HEIGHT + vh, vw + self.panel_width, 3), dtype=np.uint8)
        canvas[:] = T.BG
        canvas[T.TOP_BAR_HEIGHT:, :vw] = video
        self._top_bar(canvas, state, status)
        self._panel(canvas, vw, state, status)
        return canvas

    # -- video layers ---------------------------------------------------------
    def _video_layers(self, video: np.ndarray, frame: Optional[FrameAnalysis], state: BehaviorState,
                      status: ViewStatus, scale: float) -> None:
        cmd = state.command
        if cmd is not None and cmd.command and cmd.anchor is not None and cmd.progress > 0:
            O.command_ring(video, cmd.anchor, cmd.command, cmd.progress, scale)

        critical = [e for e in state.active_events if e.severity == "critical"]
        if critical and not state.calibrating:
            O.alert_banner(video, critical[0].message, "critical", state.t)
        elif not state.face_present and not state.calibrating:
            O.no_face_hint(video)

        O.toasts(video, self._toasts, state.t)
        O.progress_strip(video, status.progress)

        if state.calibrating:
            O.calibration_overlay(video, state.calibration_progress, state.face_present)
        if status.paused:
            O.paused_overlay(video)
        if status.show_help:
            O.help_overlay(video, status.hold_seconds)

    def _push_history(self, state: BehaviorState) -> None:
        hist = self._history
        if hist.t and state.t == hist.t[-1]:
            return  # re-render of the same frame (paused)
        if hist.t and state.t < hist.t[-1]:
            hist.t.clear()
            hist.attention.clear()
        hist.t.append(state.t)
        hist.attention.append(state.attention)
        while hist.t and hist.t[0] < state.t - self.history_seconds:
            hist.t.popleft()
            hist.attention.popleft()

    # -- chrome ---------------------------------------------------------------
    def _top_bar(self, canvas: np.ndarray, state: BehaviorState, status: ViewStatus) -> None:
        w = canvas.shape[1]
        canvas[:T.TOP_BAR_HEIGHT] = T.SURFACE
        cv2.line(canvas, (0, T.TOP_BAR_HEIGHT - 1), (w, T.TOP_BAR_HEIGHT - 1), T.BORDER, 1)
        x = 16
        x += W.text(canvas, "HBTS", (x, 27), 0.62, T.TEXT, 2, T.FONT_BOLD) + 10
        x += W.text(canvas, "Human Behavior Tracking", (x, 27), 0.45, T.MUTED, 1) + 18

        if status.recording:
            x += W.pill(canvas, "REC", x, 9, T.TEXT, T.CARD, dot=T.RECORD) + 8
        else:
            x += W.pill(canvas, "NOT RECORDING", x, 9, T.MUTED, T.CARD) + 8
        if status.privacy:
            x += W.pill(canvas, "PRIVACY", x, 9, T.BG, T.MESH) + 8
        if state.calibrating:
            x += W.pill(canvas, "CALIBRATING", x, 9, T.TEXT, T.ACCENT) + 8
        if status.paused:
            x += W.pill(canvas, "PAUSED", x, 9, T.BG, T.WARNING) + 8

        right = w - 16
        right -= W.text(canvas, f"{status.fps:4.1f} FPS", (right, 26), 0.48, T.MUTED, 1, align="right") + 18
        right -= W.text(canvas, f"{status.inference_ms:4.0f} ms", (right, 26), 0.48, T.MUTED, 1, align="right") + 18
        right -= W.text(canvas, _fmt_clock(status.elapsed), (right, 27), 0.55, T.TEXT, 1, align="right") + 18
        W.text(canvas, status.source_label, (right, 26), 0.45, T.SUBTLE, 1, align="right")

    def _panel(self, canvas: np.ndarray, x0: int, state: BehaviorState, status: ViewStatus) -> None:
        pw = self.panel_width
        top = T.TOP_BAR_HEIGHT
        H = canvas.shape[0]
        canvas[top:, x0:] = T.BG
        cv2.line(canvas, (x0, top), (x0, H), T.BORDER, 1)

        pad, gap = T.PANEL_PADDING, T.CARD_GAP
        x, cw = x0 + pad, pw - 2 * pad
        footer = 30
        avail = H - top - pad - footer - 3 * gap
        heights = [0.30, 0.23, 0.26, 0.21]
        hs = [int(avail * f) for f in heights]
        y = top + pad
        self._attention_card(canvas, x, y, cw, hs[0], state)
        y += hs[0] + gap
        self._eyes_card(canvas, x, y, cw, hs[1], state)
        y += hs[1] + gap
        self._gaze_card(canvas, x, y, cw, hs[2], state)
        y += hs[2] + gap
        self._posture_card(canvas, x, y, cw, hs[3], state)

        fy = H - 11
        kx = x
        W.text(canvas, "Press", (kx, fy), 0.42, T.SUBTLE, 1)
        kx += 46
        kx += W.keycap(canvas, "H", kx, fy - 17) + 8
        W.text(canvas, "for controls", (kx, fy), 0.42, T.SUBTLE, 1)

    # -- cards ----------------------------------------------------------------
    def _attention_card(self, img, x, y, w, h, s: BehaviorState) -> None:
        name, color = attention_label(s.attention, s.face_present, s.calibrating)
        W.card(img, x, y, w, h, "Attention", accent=color)
        score = "--" if s.calibrating else f"{int(round(s.attention))}"
        W.text(img, score, (x + 14, y + 74), 1.7, T.TEXT if not s.calibrating else T.SUBTLE, 3, T.FONT_BOLD)
        pill_fg = T.TEXT if name in ("Away", "Calibrating") else T.BG
        pill_bg = T.TRACK if name == "Away" else color
        W.pill(img, name, x + w - 14, y + 36, pill_fg, pill_bg, 0.45, align="right")
        W.text(img, f"Focus (5 min)  {s.focus_ratio * 100:3.0f}%", (x + w - 14, y + 74), 0.45, T.MUTED,
               1, align="right")

        W.hbar(img, x + 14, y + 88, w - 28, 6, s.attention / 100.0, color)
        spark_y, spark_h = y + 104, max(h - 160, 24)
        cv2.rectangle(img, (x + 14, spark_y), (x + w - 14, spark_y + spark_h), T.SURFACE, -1)
        for frac in (0.25, 0.5, 0.75):
            yy = int(spark_y + spark_h * frac)
            cv2.line(img, (x + 14, yy), (x + w - 14, yy), T.CARD, 1)
        W.sparkline(img, x + 14, spark_y, w - 28, spark_h, list(self._history.attention), 0, 100, T.ACCENT)

        fy = spark_y + spark_h + 22
        cols = (("Head", s.attention_head), ("Gaze", s.attention_gaze), ("Eyes", s.attention_eyes))
        colw = (w - 28) // 3
        for i, (label, v) in enumerate(cols):
            cx = x + 14 + i * colw
            W.label(img, label, (cx, fy))
            W.hbar(img, cx, fy + 8, colw - 14, 5, v, T.score_color(100 * v))

    def _eyes_card(self, img, x, y, w, h, s: BehaviorState) -> None:
        fatigue_color = {"Alert": T.GOOD, "Tired": T.WARNING, "Drowsy": T.CRITICAL}[s.fatigue]
        W.card(img, x, y, w, h, "Eyes & Fatigue")
        W.pill(img, s.fatigue, x + w - 14, y + 8, T.BG, fatigue_color, 0.42, align="right")

        colw = (w - 28) // 3
        face = s.face_present
        stats = (
            (f"{s.blink_count}", "Blinks"),
            (f"{s.blink_rate:.0f}", "Per min"),
            (f"{s.perclos * 100:.0f}%" if s.perclos_ready else "--", "PERCLOS"),
        )
        for i, (value, label) in enumerate(stats):
            cx = x + 14 + i * colw
            W.text(img, value, (cx, y + 62), 0.9, T.TEXT, 2, T.FONT_BOLD)
            W.label(img, label, (cx, y + 80))

        by = y + 98
        ear_max = 0.45
        bar_x, bar_w = x + 56, w - 56 - 118
        W.label(img, "EAR", (x + 14, by + 12))
        W.hbar(img, bar_x, by, bar_w, 7, s.ear_left / ear_max if face else 0.0, T.ACCENT)
        W.hbar(img, bar_x, by + 13, bar_w, 7, s.ear_right / ear_max if face else 0.0, T.ACCENT)
        if face:
            eye_state = "Closed" if s.eyes_closed else "Open"
            W.text(img, f"{s.ear:.2f}", (x + w - 70, by + 16), 0.48, T.TEXT, 1, align="right")
            W.text(img, eye_state, (x + w - 14, by + 16), 0.45, T.WARNING if s.eyes_closed else T.MUTED,
                   1, align="right")
        else:
            W.text(img, "--", (x + w - 14, by + 16), 0.48, T.SUBTLE, 1, align="right")
        if s.yawn_count or s.yawning:
            W.text(img, f"Yawns {s.yawn_count}" + ("  (now)" if s.yawning else ""),
                   (x + 14, y + h - 12), 0.42, T.MUTED, 1)

    def _gaze_card(self, img, x, y, w, h, s: BehaviorState) -> None:
        W.card(img, x, y, w, h, "Gaze & Head Pose")
        W.text(img, "relative to calibration", (x + w - 14, y + 22), 0.36, T.SUBTLE, 1, align="right")
        # Gaze map: a screen-shaped target with the combined gaze direction
        mw = int((w - 28) * 0.46)
        mh = h - 48
        mx, my = x + 14, y + 34
        W.rounded_rect(img, mx, my, mw, mh, 6, T.SURFACE)
        cx, cy = mx + mw // 2, my + mh // 2
        cv2.ellipse(img, (cx, cy), (int(mw * 0.30), int(mh * 0.30)), 0, 0, 360, T.BORDER, 1, T.AA)
        cv2.line(img, (cx - 6, cy), (cx + 6, cy), T.SUBTLE, 1, T.AA)
        cv2.line(img, (cx, cy - 6), (cx, cy + 6), T.SUBTLE, 1, T.AA)
        if s.face_present:
            # 0.30 of the half-extent corresponds to the configured tolerance (~22 deg yaw)
            gx = cx + int(np.clip(s.gaze_yaw / 22.0 * 0.30, -0.95, 0.95) * (mw / 2))
            gy = cy - int(np.clip(s.gaze_pitch / 18.0 * 0.30, -0.95, 0.95) * (mh / 2))
            dot = T.GOOD if s.on_screen else T.WARNING
            cv2.circle(img, (gx, gy), 9, dot, 2, T.AA)
            cv2.circle(img, (gx, gy), 3, dot, -1, T.AA)
        W.text(img, s.gaze_label, (cx, my + mh - 8), 0.4, T.MUTED, 1, align="center")

        # Angles
        ax = mx + mw + 16
        aw = x + w - 14 - ax
        rows = (("Yaw", s.d_yaw, 45.0), ("Pitch", s.d_pitch, 35.0), ("Roll", s.roll, 35.0))
        ry = my + 6
        for label, value, limit in rows:
            W.label(img, label, (ax, ry + 8))
            W.text(img, f"{int(round(value)):+d}", (ax + aw, ry + 9), 0.45, T.TEXT, 1, align="right")
            W.centered_bar(img, ax, ry + 16, aw, 5, value, limit, T.ACCENT)
            ry += 34
        W.label(img, "Head", (ax, ry + 10))
        W.text(img, s.head_label, (ax + aw, ry + 11), 0.48, T.TEXT, 1, align="right")

    def _posture_card(self, img, x, y, w, h, s: BehaviorState) -> None:
        W.card(img, x, y, w, h, "Posture & Hands")
        color = {"Upright": T.GOOD, "Unknown": T.TRACK}.get(s.posture, T.WARNING)
        fg = T.TEXT if s.posture == "Unknown" else T.BG
        W.pill(img, s.posture, x + w - 14, y + 8, fg, color, 0.42, align="right")
        W.text(img, s.posture_detail, (x + 14, y + 50), 0.45, T.MUTED, 1)
        if s.posture != "Unknown":
            W.text(img, f"Neck {int(round(s.neck_angle)):+d} deg    Shoulders {int(round(s.shoulder_tilt)):+d} deg",
                   (x + 14, y + 72), 0.42, T.SUBTLE, 1)

        hy = y + 98
        if not s.hands:
            W.text(img, "No hands in view", (x + 14, hy), 0.45, T.SUBTLE, 1)
            return
        for handedness, gesture, score in s.hands[:2]:
            W.text(img, handedness, (x + 14, hy), 0.48, T.TEXT, 1)
            g = "No gesture" if gesture == "None" else gesture.replace("_", " ")
            W.text(img, g, (x + 70, hy), 0.48, T.HAND if gesture != "None" else T.SUBTLE, 1)
            if gesture != "None":
                W.hbar(img, x + w - 90, hy - 8, 76, 6, score, T.HAND)
            hy += 22
