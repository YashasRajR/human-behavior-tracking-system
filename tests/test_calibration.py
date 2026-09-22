import pytest

from hbts.analytics.calibration import Calibrator


def test_baseline_is_robust_to_blinks():
    c = Calibrator(seconds=2.0, min_samples=30)
    c.start()
    t = 0.0
    done = False
    for i in range(90):
        ear = 0.05 if i % 15 == 0 else 0.30  # occasional blinks
        done = c.add(t, ear=ear, yaw=5.0, pitch=-3.0, gaze_h=0.48, gaze_v=0.45)
        t += 1 / 30
        if done:
            break
    assert done and c.baseline.calibrated
    assert c.baseline.ear == pytest.approx(0.30)
    assert c.baseline.yaw == pytest.approx(5.0)
    assert c.baseline.pitch == pytest.approx(-3.0)


def test_waits_for_face():
    c = Calibrator(seconds=1.0, min_samples=10)
    c.start()
    for i in range(100):
        assert not c.add(i / 30, ear=None)
    assert c.active
    assert c.progress(100 / 30) == 0.0


def test_requires_time_and_samples():
    c = Calibrator(seconds=1.0, min_samples=100)
    c.start()
    for i in range(60):  # 2 s but only 60 samples
        c.add(i / 30, ear=0.3)
    assert c.active
    assert 0.5 < c.progress(2.0) < 0.7


def test_restart_resets_samples():
    c = Calibrator(seconds=0.5, min_samples=5)
    c.start()
    for i in range(30):
        c.add(i / 30, ear=0.3, yaw=10.0)
    assert c.baseline.yaw == pytest.approx(10.0)
    c.start()
    assert c.active
    for i in range(30):
        c.add(10 + i / 30, ear=0.3, yaw=-4.0)
    assert c.baseline.yaw == pytest.approx(-4.0)
