import math

import numpy as np
import pytest

from hbts.core.filters import EMA, OneEuroFilter, TimeWindow


def test_one_euro_reduces_jitter_on_static_signal():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    raw, out = [], []
    for i in range(300):
        x = 10.0 + rng.normal(0, 1.0)
        raw.append(x)
        out.append(f(x, i / 30.0))
    assert np.std(out[50:]) < 0.5 * np.std(raw[50:])


def test_one_euro_tracks_fast_motion_with_beta():
    slow = OneEuroFilter(min_cutoff=0.5, beta=0.0)
    fast = OneEuroFilter(min_cutoff=0.5, beta=0.5)
    for i in range(30):
        t = i / 30.0
        value = 0.0 if i < 5 else 100.0
        a, b = slow(value, t), fast(value, t)
    assert b > a  # higher beta lags less after a step


def test_one_euro_ignores_non_increasing_timestamps():
    f = OneEuroFilter()
    f(1.0, 0.0)
    assert f(50.0, 0.0) == 1.0
    assert f(50.0, -1.0) == 1.0


def test_one_euro_rejects_bad_cutoff():
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=0)


def test_ema_time_constant():
    ema = EMA(time_constant=1.0)
    ema(0.0, 0.0)
    value = ema(1.0, 1.0)
    assert value == pytest.approx(1 - math.exp(-1))


def test_ema_is_frame_rate_independent():
    a, b = EMA(0.5), EMA(0.5)
    a(0.0, 0.0)
    b(0.0, 0.0)
    for i in range(1, 16):
        va = a(1.0, i / 15.0)
    for i in range(1, 61):
        vb = b(1.0, i / 60.0)
    assert va == pytest.approx(vb, abs=1e-9)


def test_time_window_evicts_old_samples():
    w = TimeWindow(2.0)
    for i in range(50):
        w.push(i * 0.1, 1.0 if i >= 40 else 0.0)
    assert w.span() <= 2.0
    assert w.mean() == pytest.approx(10 / 21, rel=1e-6)
    w.clear()
    assert len(w) == 0 and w.mean(default=-1) == -1
