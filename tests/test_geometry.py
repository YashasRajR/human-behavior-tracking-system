import math

import numpy as np
import pytest

from hbts.core import geometry as geo


def synthetic_eye(width=30.0, opening=9.0, cx=0.0, cy=0.0):
    """Six EAR contour points for an eye of given width and lid opening."""
    return np.array([
        [cx - width / 2, cy],
        [cx - width / 6, cy - opening / 2],
        [cx + width / 6, cy - opening / 2],
        [cx + width / 2, cy],
        [cx + width / 6, cy + opening / 2],
        [cx - width / 6, cy + opening / 2],
    ])


def rotate(points, degrees):
    a = math.radians(degrees)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return points @ r.T


def test_ear_matches_definition():
    eye = synthetic_eye(width=30, opening=9)
    assert geo.eye_aspect_ratio(eye) == pytest.approx(9 / 30)


def test_ear_is_scale_and_rotation_invariant():
    eye = synthetic_eye(width=30, opening=9)
    base = geo.eye_aspect_ratio(eye)
    assert geo.eye_aspect_ratio(eye * 3.7) == pytest.approx(base)
    assert geo.eye_aspect_ratio(rotate(eye, 25)) == pytest.approx(base)


def test_ear_drops_when_eye_closes():
    assert geo.eye_aspect_ratio(synthetic_eye(opening=1)) < geo.eye_aspect_ratio(synthetic_eye(opening=9))


def test_ear_requires_six_points():
    with pytest.raises(ValueError):
        geo.eye_aspect_ratio(np.zeros((5, 2)))


def test_mouth_aspect_ratio():
    corners = np.array([[0, 0], [50, 0]])
    closed = np.array([[[10, 0], [10, 1]], [[25, 0], [25, 1]], [[40, 0], [40, 1]]])
    wide = np.array([[[10, -15], [10, 15]], [[25, -20], [25, 20]], [[40, -15], [40, 15]]])
    assert geo.mouth_aspect_ratio(corners, closed) == pytest.approx(1 / 50)
    assert geo.mouth_aspect_ratio(corners, wide) > 0.6


def test_iris_ratio_center_and_edges():
    a, b = np.array([0.0, 0.0]), np.array([20.0, 0.0])
    top, bottom = np.array([10.0, -5.0]), np.array([10.0, 5.0])
    assert geo.iris_ratio(np.array([10.0, 0.0]), a, b, top, bottom) == pytest.approx((0.5, 0.5))
    h, _ = geo.iris_ratio(np.array([2.0, 0.0]), a, b, top, bottom)
    assert h == pytest.approx(0.1)
    h, _ = geo.iris_ratio(np.array([-50.0, 0.0]), a, b, top, bottom)
    assert h == 0.0  # clipped


def test_iris_ratio_is_roll_invariant():
    pts = np.array([[0, 0], [20, 0], [10, -5], [10, 5], [14, 1]], dtype=float)
    base = geo.iris_ratio(pts[4], pts[0], pts[1], pts[2], pts[3])
    r = rotate(pts, 30)
    rolled = geo.iris_ratio(r[4], r[0], r[1], r[2], r[3])
    assert rolled == pytest.approx(base)


def test_line_tilt_and_vertical_angle():
    assert geo.line_tilt(np.array([0, 0]), np.array([10, 0])) == pytest.approx(0)
    assert geo.line_tilt(np.array([10, 10]), np.array([0, 0])) == pytest.approx(45)
    assert geo.angle_from_vertical(np.array([0, 0]), np.array([0, 10])) == pytest.approx(0)
    assert geo.angle_from_vertical(np.array([10, 0]), np.array([0, 10])) == pytest.approx(45)


@pytest.mark.parametrize("pitch,yaw,roll", [(0, 0, 0), (12, -30, 5), (-20, 15, -8), (5, 40, 0)])
def test_euler_round_trip(pitch, yaw, roll):
    r = geo.euler_to_rotation(pitch, yaw, roll)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert geo.rotation_to_euler(r) == pytest.approx((pitch, yaw, roll), abs=1e-6)


@pytest.mark.parametrize("pitch,yaw,roll", [(0, 20, 0), (15, 0, 0), (0, 0, 10), (10, -25, 5)])
def test_head_pose_recovers_synthetic_projection(pitch, yaw, roll):
    cv2 = pytest.importorskip("cv2")
    k = geo.camera_matrix(640, 480)
    rvec, _ = cv2.Rodrigues(geo.euler_to_rotation(pitch, yaw, roll))
    pts, _ = cv2.projectPoints(geo.HEAD_MODEL_3D, rvec, np.array([20.0, -10.0, 1500.0]), k, np.zeros(4))
    est = geo.estimate_head_pose(pts.reshape(-1, 2), 640, 480)
    assert est == pytest.approx((pitch, yaw, roll), abs=0.5)


def test_yaw_sign_convention():
    """Positive yaw must mean the nose points towards image-right."""
    cv2 = pytest.importorskip("cv2")
    k = geo.camera_matrix(640, 480)
    rvec, _ = cv2.Rodrigues(geo.euler_to_rotation(0, 25, 0))
    pts, _ = cv2.projectPoints(geo.HEAD_MODEL_3D, rvec, np.array([0.0, 0.0, 1500.0]), k, np.zeros(4))
    pts = pts.reshape(-1, 2)
    nose_x, eyes_mid_x = pts[0, 0], pts[2:4, 0].mean()
    assert nose_x > eyes_mid_x


def test_wrap_degrees():
    assert geo.wrap_degrees(190) == pytest.approx(-170)
    assert geo.wrap_degrees(-180) == pytest.approx(180)
    assert geo.wrap_degrees(45) == pytest.approx(45)


def test_head_pose_accepts_non_contiguous_float64_input():
    pytest.importorskip("cv2")
    k = geo.camera_matrix(640, 480)
    import cv2

    rvec, _ = cv2.Rodrigues(geo.euler_to_rotation(5, 10, 0))
    pts, _ = cv2.projectPoints(geo.HEAD_MODEL_3D, rvec, np.array([0.0, 0.0, 1500.0]), k, np.zeros(4))
    padded = np.c_[pts.reshape(-1, 2), np.zeros(6)].astype(np.float64)  # (6, 3)
    view = padded[:, :2]  # non-contiguous view
    assert not view.flags["C_CONTIGUOUS"]
    assert geo.estimate_head_pose(view, 640, 480) == pytest.approx((5, 10, 0), abs=0.5)
