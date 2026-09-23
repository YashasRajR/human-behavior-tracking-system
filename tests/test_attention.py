import pytest

from hbts.analytics.attention import AttentionScorer, falloff


def settle(scorer, seconds=5.0, **kwargs):
    update = None
    for i in range(int(seconds * 30)):
        update = scorer.update(i / 30.0, **kwargs)
    return update


def test_falloff_halves_at_tolerance():
    assert falloff(0.0) == pytest.approx(1.0)
    assert falloff(1.0) == pytest.approx(0.5)
    assert falloff(0.6, 0.8) == pytest.approx(0.5)


def test_frontal_face_scores_high():
    u = settle(AttentionScorer(), face_present=True)
    assert u.score == pytest.approx(100.0, abs=0.5)
    assert u.on_screen


def test_head_turn_lowers_score():
    u = settle(AttentionScorer(), face_present=True, d_yaw=45.0)
    assert u.score < 40
    assert not u.on_screen


def test_eye_counter_rotation_keeps_gaze_on_screen():
    """Head turned right while the eyes look back at the screen."""
    scorer = AttentionScorer(yaw_tolerance=22.0, gaze_tolerance=0.16)
    u = settle(scorer, face_present=True, d_yaw=20.0, d_gaze_h=-0.145)
    assert u.gaze > 0.95
    assert u.on_screen
    assert u.head < u.gaze


def test_eye_rotation_in_same_direction_is_worse():
    scorer = AttentionScorer()
    away = settle(scorer, face_present=True, d_yaw=15.0, d_gaze_h=0.10)
    assert away.gaze < 0.5


def test_absent_face_decays_to_zero():
    scorer = AttentionScorer(time_constant=0.5)
    settle(scorer, face_present=True)
    u = None
    for i in range(150, 300):
        u = scorer.update(i / 30.0, face_present=False)
    assert u.score < 2.0


def test_sustained_eye_closure_reduces_score():
    open_ = settle(AttentionScorer(), face_present=True)
    closed = settle(AttentionScorer(), face_present=True, eyes_closed_long=True)
    assert closed.score < open_.score - 15
    assert not closed.on_screen


def test_weights_are_normalised():
    s = AttentionScorer(weights={"head": 2, "gaze": 1, "eyes": 1})
    assert s.w_head + s.w_gaze + s.w_eyes == pytest.approx(1.0)
