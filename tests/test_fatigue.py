import pytest

from hbts.analytics.fatigue import PerclosMeter, YawnDetector, fatigue_level


def test_perclos_fraction():
    m = PerclosMeter(window=10.0, warmup=5.0)
    for i in range(300):  # 10 s at 30 FPS, eyes closed 25% of the time
        m.update(i % 4 == 0, i / 30.0)
    assert m.value == pytest.approx(0.25, abs=0.01)
    assert m.ready


def test_perclos_not_ready_before_warmup():
    m = PerclosMeter(window=60.0, warmup=15.0)
    for i in range(60):
        m.update(False, i / 30.0)
    assert not m.ready


def test_yawn_requires_sustained_opening():
    y = YawnDetector(mar_threshold=0.5, min_seconds=1.0)
    t = 0.0
    # Speech-like short openings
    for _ in range(10):
        for mar in (0.6, 0.6, 0.2, 0.2):
            y.update(mar, t)
            t += 1 / 30
    assert y.total == 0
    # A real yawn
    started = []
    for _ in range(60):
        started.append(y.update(0.8, t).started)
        t += 1 / 30
    assert y.total == 1
    assert sum(started) == 1


def test_fatigue_levels():
    assert fatigue_level(0.02, 0.15, 0, False) == "Alert"
    assert fatigue_level(0.18, 0.15, 0, False) == "Tired"
    assert fatigue_level(0.05, 0.15, 3, False) == "Tired"
    assert fatigue_level(0.35, 0.15, 0, False) == "Drowsy"
    assert fatigue_level(0.0, 0.15, 0, True) == "Drowsy"
