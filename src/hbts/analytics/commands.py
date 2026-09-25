"""Hands-free control: map held gestures to application commands.

A gesture only triggers its command after being held continuously for
``hold_seconds``. This dwell-time pattern (common in gaze- and gesture-based
interfaces) prevents accidental triggers from gestures that appear briefly
during natural hand movement. After firing, the same gesture must be released
before it can fire again, and a global cool-down applies.

The current hold progress is exposed so the UI can draw a filling ring,
giving the user immediate feedback that a command is about to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from hbts.core.types import HandObservation


@dataclass
class CommandState:
    gesture: Optional[str]
    command: Optional[str]
    progress: float                 # 0..1 hold progress
    anchor: Optional[Tuple[int, int]]  # where to draw the progress ring
    fired: Optional[str] = None     # command fired on this update


class GestureCommandMapper:
    def __init__(
        self,
        bindings: Dict[str, str],
        hold_seconds: float = 1.2,
        cooldown_seconds: float = 2.0,
        min_score: float = 0.6,
    ) -> None:
        self.bindings = dict(bindings)
        self.hold_seconds = hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.min_score = min_score
        self._gesture: Optional[str] = None
        self._since: Optional[float] = None
        self._latched = False
        self._last_fire = -1e9

    def _candidate(self, hands: Iterable[HandObservation]) -> Optional[HandObservation]:
        best: Optional[HandObservation] = None
        for hand in hands:
            if hand.gesture in self.bindings and hand.gesture_score >= self.min_score:
                if best is None or hand.gesture_score > best.gesture_score:
                    best = hand
        return best

    def update(self, hands: Iterable[HandObservation], t: float) -> CommandState:
        hand = self._candidate(hands)
        gesture = hand.gesture if hand is not None else None

        if gesture != self._gesture:
            self._gesture = gesture
            self._since = t if gesture is not None else None
            self._latched = False

        if gesture is None or self._since is None:
            return CommandState(None, None, 0.0, None)

        command = self.bindings[gesture]
        anchor = hand.center if hand is not None else None
        if self._latched:
            return CommandState(gesture, command, 1.0, anchor)

        progress = min((t - self._since) / self.hold_seconds, 1.0) if self.hold_seconds > 0 else 1.0
        cooling = (t - self._last_fire) < self.cooldown_seconds
        if progress >= 1.0 and not cooling:
            self._latched = True
            self._last_fire = t
            return CommandState(gesture, command, 1.0, anchor, fired=command)
        if cooling:
            progress = 0.0
            self._since = t
        return CommandState(gesture, command, progress, anchor)
