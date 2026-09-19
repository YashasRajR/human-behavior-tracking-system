import numpy as np
import pytest

from hbts.vision.face import extract_face_features
from hbts.vision.pose import extract_pose_features


def test_features_on_recorded_face(frontal_landmarks):
    lm, w, h = frontal_landmarks
    face = extract_face_features(lm, w, h)
    assert 0.2 < face.ear_left < 0.6 and 0.2 < face.ear_right < 0.6
    assert face.mar < 0.1  # mouth closed
    assert 0.0 <= face.gaze_h <= 1.0 and 0.0 <= face.gaze_v <= 1.0
    x, y, bw, bh = face.bbox
    assert 0 <= x < w and 0 <= y < h and bw > 50 and bh > 50
    assert abs(face.roll) < 15


def test_mirroring_flips_yaw_and_gaze(frontal_landmarks):
    lm, w, h = frontal_landmarks
    mirrored = lm.copy()
    mirrored[:, 0] = (w - 1) - mirrored[:, 0]
    a = extract_face_features(lm, w, h)
    b = extract_face_features(mirrored, w, h)
    assert b.yaw == pytest.approx(-a.yaw, abs=2.0)
    assert b.pitch == pytest.approx(a.pitch, abs=2.0)
    assert b.gaze_h == pytest.approx(1.0 - a.gaze_h, abs=0.03)


def test_requires_refined_mesh():
    with pytest.raises(ValueError):
        extract_face_features(np.zeros((468, 3)), 640, 480)


def _pose(nose_y=100.0, tilt=0.0):
    lm = np.zeros((33, 3), dtype=np.float32)
    lm[0] = [200, nose_y, 0]
    lm[7] = [230, nose_y - 5, 0]
    lm[8] = [170, nose_y - 5, 0]
    lm[11] = [300, 250 + tilt, 0]
    lm[12] = [100, 250 - tilt, 0]
    return lm, np.ones(33, dtype=np.float32)


def test_pose_features():
    p = extract_pose_features(*_pose())
    assert p.shoulder_tilt == pytest.approx(0.0, abs=1e-6)
    assert abs(p.neck_angle) < 1e-6
    assert p.head_drop == pytest.approx(150 / 200)
    slouched = extract_pose_features(*_pose(nose_y=180.0))
    assert slouched.head_drop < p.head_drop


def test_pose_requires_visible_shoulders():
    lm, vis = _pose()
    vis[11] = 0.1
    assert extract_pose_features(lm, vis) is None
