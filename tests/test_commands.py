import numpy as np

from hbts.analytics.commands import GestureCommandMapper
from hbts.core.types import HandObservation


def hand(gesture, score=0.9):
    return HandObservation("Right", gesture, score, np.zeros((21, 3), np.float32), (100, 100, 50, 80))


def run(mapper, frames, t0=0.0, fps=30.0):
    fired = []
    t = t0
    for hands in frames:
        state = mapper.update(hands, t)
        if state.fired:
            fired.append((round(t, 2), state.fired))
        t += 1 / fps
    return fired, t


def test_fires_once_after_hold():
    m = GestureCommandMapper({"Thumb_Up": "bookmark"}, hold_seconds=1.0, cooldown_seconds=0.0)
    fired, _ = run(m, [[hand("Thumb_Up")]] * 90)
    assert len(fired) == 1
    assert fired[0][1] == "bookmark"
    assert 0.95 <= fired[0][0] <= 1.1


def test_brief_gesture_does_not_fire():
    m = GestureCommandMapper({"Victory": "snapshot"}, hold_seconds=1.0)
    fired, _ = run(m, [[hand("Victory")]] * 20 + [[]] * 10 + [[hand("Victory")]] * 20)
    assert fired == []


def test_release_and_cooldown():
    m = GestureCommandMapper({"Thumb_Up": "bookmark"}, hold_seconds=0.5, cooldown_seconds=3.0)
    frames = [[hand("Thumb_Up")]] * 30 + [[]] * 5 + [[hand("Thumb_Up")]] * 30
    fired, t = run(m, frames)
    assert len(fired) == 1  # second hold falls inside the cool-down
    fired2, _ = run(m, [[]] * 5 + [[hand("Thumb_Up")]] * 30, t0=t + 5.0)
    assert len(fired2) == 1


def test_low_confidence_and_unbound_gestures_are_ignored():
    m = GestureCommandMapper({"Thumb_Up": "bookmark"}, hold_seconds=0.2, min_score=0.7)
    fired, _ = run(m, [[hand("Thumb_Up", 0.5), hand("Pointing_Up")]] * 30)
    assert fired == []


def test_progress_is_reported():
    m = GestureCommandMapper({"Open_Palm": "toggle_privacy"}, hold_seconds=1.0)
    state = None
    for i in range(15):
        state = m.update([hand("Open_Palm")], i / 30)
    assert state.command == "toggle_privacy"
    assert 0.4 < state.progress < 0.6
    assert state.anchor == (125, 140)
