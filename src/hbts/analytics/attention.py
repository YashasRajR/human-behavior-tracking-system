"""Attention score: a continuous 0-100 estimate of on-screen engagement.

The score fuses three cues, each mapped to ``[0, 1]``:

1. **Head orientation** - deviation of yaw/pitch from the calibrated neutral
   pose, passed through a Gaussian falloff that halves the factor at the
   configured tolerance.

2. **Gaze direction** - where the eyes are actually pointing. People routinely
   turn their head slightly while keeping their eyes on the screen (the
   vestibulo-ocular reflex counter-rotates the eyes). Treating head pose alone
   as attention would penalise that, so the eye-in-head rotation estimated from
   the iris position is *added* to the head angle before applying the falloff:

       gaze_yaw   = d_yaw   + K * d_iris_h
       gaze_pitch = d_pitch - K * d_iris_v

   ``K`` (degrees per unit of iris ratio) converts the normalised iris offset
   to an approximate eye rotation.

3. **Eye openness** - 1 while the eyes are open, 0 during sustained closure
   (brief blinks are ignored by the output smoothing).

``score = 100 * (w_head * head + w_gaze * gaze + w_eyes * eyes)``, followed by
an exponential moving average with a time constant in seconds. When no face is
visible every factor is zero and the score decays smoothly to zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from hbts.core.filters import EMA

LN2 = math.log(2.0)


def falloff(*normalised: float) -> float:
    """Gaussian falloff: 1 at zero deviation, 0.5 when the combined deviation equals 1."""
    return math.exp(-LN2 * sum(x * x for x in normalised))


@dataclass
class AttentionUpdate:
    score: float           # smoothed, 0-100
    raw: float             # unsmoothed, 0-100
    head: float
    gaze: float
    eyes: float
    on_screen: bool
    gaze_yaw: float        # combined head+eye direction relative to neutral, degrees
    gaze_pitch: float


class AttentionScorer:
    def __init__(
        self,
        yaw_tolerance: float = 22.0,
        pitch_tolerance: float = 18.0,
        gaze_tolerance: float = 0.16,
        weights: Optional[Dict[str, float]] = None,
        time_constant: float = 0.8,
        eye_gain: Optional[float] = None,
    ) -> None:
        self.yaw_tol = yaw_tolerance
        self.pitch_tol = pitch_tolerance
        # Degrees of eye rotation per unit of iris ratio. By default an iris
        # offset equal to ``gaze_tolerance`` is treated like a head turn equal
        # to ``yaw_tolerance``.
        self.eye_gain = eye_gain if eye_gain is not None else yaw_tolerance / gaze_tolerance
        w = weights or {"head": 0.45, "gaze": 0.35, "eyes": 0.20}
        total = sum(w.values()) or 1.0
        self.w_head = w.get("head", 0.0) / total
        self.w_gaze = w.get("gaze", 0.0) / total
        self.w_eyes = w.get("eyes", 0.0) / total
        self._ema = EMA(time_constant)
        self._eyes_ema = EMA(0.35)

    def reset(self) -> None:
        self._ema.reset()
        self._eyes_ema.reset()

    def update(
        self,
        t: float,
        face_present: bool,
        d_yaw: float = 0.0,
        d_pitch: float = 0.0,
        d_gaze_h: float = 0.0,
        d_gaze_v: float = 0.0,
        eyes_closed_long: bool = False,
    ) -> AttentionUpdate:
        if not face_present:
            raw = 0.0
            score = self._ema(raw, t)
            self._eyes_ema(0.0, t)
            return AttentionUpdate(score, raw, 0.0, 0.0, 0.0, False, 0.0, 0.0)

        head = falloff(d_yaw / self.yaw_tol, d_pitch / self.pitch_tol)
        gaze_yaw = d_yaw + self.eye_gain * d_gaze_h
        gaze_pitch = d_pitch - self.eye_gain * d_gaze_v
        gaze = falloff(gaze_yaw / self.yaw_tol, gaze_pitch / self.pitch_tol)
        eyes = self._eyes_ema(0.0 if eyes_closed_long else 1.0, t)

        raw = 100.0 * (self.w_head * head + self.w_gaze * gaze + self.w_eyes * eyes)
        score = self._ema(raw, t)
        on_screen = gaze >= 0.5 and not eyes_closed_long
        return AttentionUpdate(score, raw, head, gaze, eyes, on_screen, gaze_yaw, gaze_pitch)
