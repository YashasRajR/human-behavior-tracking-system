"""Behavior engine: turns per-frame perception into interpretable state.

``BehaviorEngine.update`` is the single entry point used by the application.
It owns every analytics component, applies calibration baselines, evaluates
event rules and produces a ``BehaviorState`` snapshot that the dashboard and
the session recorder consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hbts.analytics.attention import AttentionScorer
from hbts.analytics.blink import BlinkDetector
from hbts.analytics.calibration import Calibrator
from hbts.analytics.commands import CommandState, GestureCommandMapper
from hbts.analytics.events import (
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    Event,
    EventEngine,
)
from hbts.analytics.fatigue import PerclosMeter, YawnDetector, fatigue_level
from hbts.core.filters import TimeWindow
from hbts.core.types import FrameAnalysis

HEAD_YAW_LABEL = 15.0
HEAD_PITCH_LABEL = 12.0

EVENT_LABELS = {
    "distracted": "Looking away from screen",
    "absent": "No face in view",
    "low_attention": "Sustained low attention",
    "microsleep": "Eyes closed (microsleep)",
    "drowsy": "Drowsiness detected (PERCLOS)",
    "poor_posture": "Poor posture",
    "yawn": "Yawn",
    "bookmark": "Bookmark",
    "snapshot": "Snapshot saved",
    "calibrated": "Calibration complete",
    "command": "Gesture command",
}


@dataclass
class BehaviorState:
    t: float = 0.0
    face_present: bool = False
    calibrating: bool = True
    calibration_progress: float = 0.0

    ear: float = 0.0
    ear_left: float = 0.0
    ear_right: float = 0.0
    eyes_closed: bool = False
    closure_seconds: float = 0.0
    blink_count: int = 0
    blink_rate: float = 0.0
    blinked: bool = False

    perclos: float = 0.0
    perclos_ready: bool = False
    mar: float = 0.0
    yawning: bool = False
    yawn_count: int = 0
    fatigue: str = "Alert"

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    d_yaw: float = 0.0
    d_pitch: float = 0.0
    head_label: str = "-"
    gaze_h: float = 0.5
    gaze_v: float = 0.5
    gaze_yaw: float = 0.0
    gaze_pitch: float = 0.0
    gaze_label: str = "-"

    attention: float = 0.0
    attention_head: float = 0.0
    attention_gaze: float = 0.0
    attention_eyes: float = 0.0
    on_screen: bool = False
    focus_ratio: float = 0.0

    posture: str = "Unknown"
    posture_detail: str = ""
    shoulder_tilt: float = 0.0
    neck_angle: float = 0.0

    hands: List[Tuple[str, str, float]] = field(default_factory=list)
    command: Optional[CommandState] = None

    active_events: List[Event] = field(default_factory=list)
    new_events: List[Event] = field(default_factory=list)

    def metrics(self) -> Dict[str, Any]:
        """Flat, JSON-serialisable subset written by the session recorder."""
        return {
            "t": round(self.t, 3),
            "face": self.face_present,
            "calibrating": self.calibrating,
            "attention": round(self.attention, 2),
            "on_screen": self.on_screen,
            "ear": round(self.ear, 4),
            "eyes_closed": self.eyes_closed,
            "blink_count": self.blink_count,
            "blink_rate": round(self.blink_rate, 2),
            "perclos": round(self.perclos, 4),
            "perclos_ready": self.perclos_ready,
            "mar": round(self.mar, 4),
            "yawn_count": self.yawn_count,
            "fatigue": self.fatigue,
            "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2),
            "roll": round(self.roll, 2),
            "gaze_h": round(self.gaze_h, 4),
            "gaze_v": round(self.gaze_v, 4),
            "head": self.head_label,
            "gaze": self.gaze_label,
            "posture": self.posture,
            "gestures": [g for _, g, _ in self.hands if g != "None"],
        }


def _direction_label(d_yaw: float, d_pitch: float, yaw_limit: float, pitch_limit: float) -> str:
    horizontal = "Right" if d_yaw > yaw_limit else "Left" if d_yaw < -yaw_limit else ""
    vertical = "Up" if d_pitch > pitch_limit else "Down" if d_pitch < -pitch_limit else ""
    if horizontal and vertical:
        return f"{vertical}-{horizontal}"
    return horizontal or vertical or "Center"


class BehaviorEngine:
    def __init__(self, cfg: Dict[str, Any], on_event=None) -> None:
        self.cfg = cfg
        b, f, a, c = cfg["blink"], cfg["fatigue"], cfg["attention"], cfg["calibration"]
        self.blinks = BlinkDetector(
            close_ratio=b["close_ratio"], open_ratio=b["open_ratio"],
            min_duration=b["min_duration"], max_duration=b["max_duration"],
            rate_window=b["rate_window"],
        )
        self.perclos = PerclosMeter(window=f["perclos_window"])
        self.yawns = YawnDetector(mar_threshold=f["yawn_mar"], min_seconds=f["yawn_seconds"])
        self.calibrator = Calibrator(seconds=c["seconds"], min_samples=c["min_samples"], default_ear=b["default_ear"])
        self.attention = AttentionScorer(
            yaw_tolerance=a["yaw_tolerance"], pitch_tolerance=a["pitch_tolerance"],
            gaze_tolerance=a["gaze_tolerance"], weights=a["weights"], time_constant=a["time_constant"],
        )
        cmd = cfg["commands"]
        self.commands = (
            GestureCommandMapper(cmd["bindings"], cmd["hold_seconds"], cmd["cooldown_seconds"], cmd["min_score"])
            if cmd.get("enabled", True) else None
        )

        ev = cfg["events"]
        self.events = EventEngine(on_event=on_event)
        self.events.add_rule("absent", ev["absent_seconds"], 1.0, SEVERITY_WARNING, EVENT_LABELS["absent"])
        self.events.add_rule("distracted", ev["distracted_seconds"], 0.8, SEVERITY_WARNING, EVENT_LABELS["distracted"])
        self.events.add_rule("low_attention", 10.0, 3.0, SEVERITY_WARNING, EVENT_LABELS["low_attention"])
        self.events.add_rule("microsleep", f["microsleep_seconds"], 0.2, SEVERITY_CRITICAL, EVENT_LABELS["microsleep"])
        self.events.add_rule("drowsy", 5.0, 10.0, SEVERITY_CRITICAL, EVENT_LABELS["drowsy"])
        self.events.add_rule("poor_posture", ev["slouch_seconds"], 2.0, SEVERITY_WARNING, EVENT_LABELS["poor_posture"])

        self._focus = TimeWindow(300.0)
        self._t0: Optional[float] = None
        self.calibrator.start()

    # -- helpers --------------------------------------------------------------
    def recalibrate(self) -> None:
        self.calibrator.start()

    def mark(self, kind: str, t: float, message: str = "", data: Optional[Dict[str, Any]] = None) -> Event:
        return self.events.instant(kind, t, message or EVENT_LABELS.get(kind, kind), SEVERITY_INFO, data)

    def _posture(self, pose, state: BehaviorState) -> None:
        if pose is None:
            state.posture, state.posture_detail = "Unknown", "Upper body not visible"
            return
        p = self.cfg["posture"]
        base = self.calibrator.baseline
        state.shoulder_tilt = pose.shoulder_tilt
        state.neck_angle = pose.neck_angle - base.neck_angle

        issues = []
        if base.head_drop and pose.head_drop < p["head_drop_ratio"] * base.head_drop:
            issues.append(("Slouching", "Head has dropped towards the shoulders"))
        if abs(state.neck_angle) > p["neck_angle_limit"]:
            issues.append(("Leaning", "Head tilted to one side"))
        if abs(pose.shoulder_tilt) > p["shoulder_tilt_limit"]:
            issues.append(("Uneven", "Shoulders are not level"))
        if issues:
            state.posture, state.posture_detail = issues[0]
        else:
            state.posture, state.posture_detail = "Upright", "Neutral posture"

    # -- main update ----------------------------------------------------------
    def update(self, frame: FrameAnalysis) -> BehaviorState:
        t = frame.t
        if self._t0 is None:
            self._t0 = t
        s = BehaviorState(t=t)
        face, pose = frame.face, frame.pose

        # Calibration sampling
        if self.calibrator.active:
            done = self.calibrator.add(
                t,
                ear=face.ear if face else None,
                yaw=face.yaw if face else None,
                pitch=face.pitch if face else None,
                gaze_h=face.gaze_h if face else None,
                gaze_v=face.gaze_v if face else None,
                head_drop=pose.head_drop if pose else None,
                neck_angle=pose.neck_angle if pose else None,
            )
            if done:
                self.attention.reset()
                self.perclos.reset()
                s.new_events.append(self.mark("calibrated", t))
        s.calibrating = self.calibrator.active
        s.calibration_progress = self.calibrator.progress(t)
        base = self.calibrator.baseline

        # Face-derived metrics
        s.face_present = face is not None
        closed_long = False
        if face is not None:
            s.ear, s.ear_left, s.ear_right = face.ear, face.ear_left, face.ear_right
            blink = self.blinks.update(face.ear, t, base.ear)
            s.eyes_closed, s.closure_seconds, s.blinked = blink.closed, blink.closure_seconds, blink.blinked
            closed_long = blink.closure_seconds >= 0.4
            s.perclos = self.perclos.update(blink.closed, t)

            s.mar = face.mar
            yawn = self.yawns.update(face.mar, t)
            s.yawning = yawn.yawning
            if yawn.started:
                s.new_events.append(self.mark("yawn", t))

            s.yaw, s.pitch, s.roll = face.yaw, face.pitch, face.roll
            s.d_yaw, s.d_pitch = face.yaw - base.yaw, face.pitch - base.pitch
            s.gaze_h, s.gaze_v = face.gaze_h, face.gaze_v
            s.head_label = _direction_label(s.d_yaw, s.d_pitch, HEAD_YAW_LABEL, HEAD_PITCH_LABEL)
        else:
            self.blinks.mark_absent()

        s.blink_count = self.blinks.total
        s.blink_rate = self.blinks.rate_per_minute()
        s.perclos_ready = self.perclos.ready
        s.yawn_count = self.yawns.total

        att = self.attention.update(
            t, s.face_present, s.d_yaw, s.d_pitch,
            s.gaze_h - base.gaze_h, s.gaze_v - base.gaze_v, closed_long,
        )
        s.attention, s.on_screen = att.score, att.on_screen
        s.attention_head, s.attention_gaze, s.attention_eyes = att.head, att.gaze, att.eyes
        s.gaze_yaw, s.gaze_pitch = att.gaze_yaw, att.gaze_pitch
        if s.face_present:
            s.gaze_label = _direction_label(att.gaze_yaw, att.gaze_pitch, 20.0, 15.0)

        microsleep = s.closure_seconds >= self.cfg["fatigue"]["microsleep_seconds"]
        s.fatigue = fatigue_level(
            s.perclos if s.perclos_ready else 0.0,
            self.cfg["fatigue"]["perclos_threshold"], self.yawns.recent(), microsleep,
        )

        self._posture(pose, s)
        s.hands = [(h.handedness, h.gesture, h.gesture_score) for h in frame.hands]

        if self.commands is not None:
            s.command = self.commands.update(frame.hands, t)

        # Focus ratio over the trailing five minutes (excluding calibration)
        if not s.calibrating:
            self._focus.push(t, 1.0 if s.on_screen else 0.0)
        s.focus_ratio = self._focus.mean(0.0)

        # Event rules are suspended while calibrating
        if not s.calibrating:
            conditions = {
                "absent": not s.face_present,
                "distracted": s.face_present and not s.on_screen and not s.eyes_closed,
                "low_attention": s.face_present and s.attention < self.cfg["events"]["low_attention"],
                "microsleep": microsleep,
                "drowsy": s.perclos_ready and s.perclos >= 2.0 * self.cfg["fatigue"]["perclos_threshold"],
                "poor_posture": s.posture in ("Slouching", "Leaning", "Uneven"),
            }
            s.new_events.extend(self.events.update(t, conditions))
        s.active_events = self.events.active()
        return s

    def summary(self, t: float) -> Dict[str, Any]:
        duration = 0.0 if self._t0 is None else t - self._t0
        return {
            "duration_seconds": round(duration, 2),
            "blinks": self.blinks.total,
            "long_eye_closures": self.blinks.long_closures,
            "yawns": self.yawns.total,
            "baseline": self.calibrator.baseline.to_dict(),
            "events": self.events.totals(t),
        }
