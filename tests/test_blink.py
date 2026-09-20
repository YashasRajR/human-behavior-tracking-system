import numpy as np
import pytest

from hbts.analytics.blink import BlinkDetector

FPS = 30.0
BASE = 0.30


def run(detector, segments, noise=0.0, seed=0):
    """segments: list of (seconds, ear). Returns blink count."""
    rng = np.random.default_rng(seed)
    t = 0.0
    for seconds, ear in segments:
        for _ in range(int(round(seconds * FPS))):
            detector.update(ear + rng.normal(0, noise), t, BASE)
            t += 1.0 / FPS
    return detector.total


def test_counts_normal_blinks():
    d = BlinkDetector()
    segments = []
    for _ in range(5):
        segments += [(1.0, 0.30), (0.15, 0.08)]
    segments.append((0.5, 0.30))
    assert run(d, segments) == 5


def test_long_closure_is_not_a_blink():
    d = BlinkDetector(max_duration=0.5)
    assert run(d, [(1.0, 0.30), (1.5, 0.05), (1.0, 0.30)]) == 0
    assert d.long_closures == 1


def test_single_frame_dip_is_ignored():
    d = BlinkDetector(min_duration=0.05)
    assert run(d, [(1.0, 0.30), (1 / FPS, 0.05), (1.0, 0.30)]) == 0


def test_hysteresis_prevents_double_counting():
    # EAR hovering between the close (0.216) and open (0.246) thresholds after
    # a closure must not register additional blinks.
    d = BlinkDetector(close_ratio=0.72, open_ratio=0.82)
    segments = [(1.0, 0.30), (0.10, 0.10), (0.15, 0.23), (0.05, 0.20), (0.05, 0.23), (1.0, 0.30)]
    assert run(d, segments) == 1


def test_adapts_to_personal_baseline():
    # A person with a naturally narrow eye (EAR 0.20) would look permanently
    # "closed" to a 0.22 fixed threshold; relative thresholds handle it.
    d = BlinkDetector()
    t = 0.0
    for i in range(90):
        d.update(0.20 if i % 30 not in (10, 11, 12, 13) else 0.05, t, baseline=0.20)
        t += 1 / FPS
    assert d.total == 3


def test_rate_per_minute():
    d = BlinkDetector(rate_window=60.0)
    segments = []
    for _ in range(10):  # one blink every 3 s -> 20 / min
        segments += [(2.85, 0.30), (0.15, 0.08)]
    run(d, segments)
    assert d.rate_per_minute() == pytest.approx(20.0, rel=0.1)


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        BlinkDetector(close_ratio=0.9, open_ratio=0.8)
