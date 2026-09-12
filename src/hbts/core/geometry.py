"""Geometric feature extraction from facial and body landmarks.

All functions operate on NumPy arrays in pixel space and are free of any
MediaPipe state, which keeps them deterministic and unit-testable.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def distance(p: np.ndarray, q: np.ndarray) -> float:
    """Euclidean distance between two 2-D (or 3-D) points."""
    return float(np.linalg.norm(np.asarray(p, dtype=float) - np.asarray(q, dtype=float)))


def eye_aspect_ratio(points: np.ndarray) -> float:
    """Eye Aspect Ratio (EAR) from six contour points.

    ``EAR = (|p2 - p6| + |p3 - p5|) / (2 |p1 - p4|)``

    The ratio is roughly constant while the eye is open and falls towards zero
    during a blink, independent of face scale and in-plane rotation.

    Reference: T. Soukupova and J. Cech, "Real-Time Eye Blink Detection using
    Facial Landmarks", CVWW 2016.
    """
    p = np.asarray(points, dtype=float)
    if p.shape[0] != 6:
        raise ValueError("eye_aspect_ratio expects exactly six points")
    horizontal = distance(p[0], p[3])
    if horizontal < 1e-6:
        return 0.0
    vertical = distance(p[1], p[5]) + distance(p[2], p[4])
    return vertical / (2.0 * horizontal)


def mouth_aspect_ratio(corners: np.ndarray, vertical_pairs: np.ndarray) -> float:
    """Mouth Aspect Ratio (MAR): mean inner-lip opening over mouth width.

    ``corners`` is a (2, D) array; ``vertical_pairs`` is (K, 2, D).
    """
    c = np.asarray(corners, dtype=float)
    v = np.asarray(vertical_pairs, dtype=float)
    width = distance(c[0], c[1])
    if width < 1e-6:
        return 0.0
    openings = [distance(a, b) for a, b in v]
    return float(np.mean(openings)) / width


def iris_ratio(
    iris: np.ndarray,
    corner_a: np.ndarray,
    corner_b: np.ndarray,
    lid_top: np.ndarray,
    lid_bottom: np.ndarray,
) -> Tuple[float, float]:
    """Normalised iris position inside the eye opening.

    The iris centre is projected onto the axis running from ``corner_a`` to
    ``corner_b`` (horizontal ratio) and onto the lid axis (vertical ratio).
    Projecting onto the eye's own axes, rather than using raw image x/y, makes
    the measure robust to head roll.

    Returns ``(h, v)`` where ``h = 0`` at ``corner_a`` and ``1`` at ``corner_b``,
    and ``v = 0`` at the upper lid and ``1`` at the lower lid. Values are
    clipped to ``[0, 1]``.
    """
    iris = np.asarray(iris, dtype=float)[:2]
    a = np.asarray(corner_a, dtype=float)[:2]
    b = np.asarray(corner_b, dtype=float)[:2]
    top = np.asarray(lid_top, dtype=float)[:2]
    bottom = np.asarray(lid_bottom, dtype=float)[:2]

    axis = b - a
    length_sq = float(axis @ axis)
    h = 0.5 if length_sq < 1e-9 else float((iris - a) @ axis) / length_sq

    lid_axis = bottom - top
    lid_sq = float(lid_axis @ lid_axis)
    v = 0.5 if lid_sq < 1e-9 else float((iris - top) @ lid_axis) / lid_sq

    return float(np.clip(h, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))


def angle_from_vertical(top: np.ndarray, bottom: np.ndarray) -> float:
    """Angle in degrees between the segment ``bottom -> top`` and image-up.

    0 means perfectly vertical; the sign is positive when ``top`` leans towards
    image-right.
    """
    dx = float(top[0] - bottom[0])
    dy = float(bottom[1] - top[1])  # image y grows downwards
    return math.degrees(math.atan2(dx, dy))


def line_tilt(p: np.ndarray, q: np.ndarray) -> float:
    """Tilt of the line through ``p`` and ``q`` relative to the image horizontal.

    Returned in degrees within (-90, 90]; positive when the right-hand end of
    the line is lower in the image.
    """
    dx = float(q[0] - p[0])
    dy = float(q[1] - p[1])
    if dx < 0:
        dx, dy = -dx, -dy
    return math.degrees(math.atan2(dy, dx))


# ---------------------------------------------------------------------------
# Head pose
# ---------------------------------------------------------------------------

# Generic 3-D face model (millimetres) in *camera* convention:
# x to image-right, y to image-down, z away from the camera.
# Order matches ``landmarks.HEAD_POSE_ANCHORS``.
HEAD_MODEL_3D = np.array(
    [
        [0.0, 0.0, 0.0],          # nose tip
        [0.0, 330.0, 65.0],       # chin
        [-225.0, -170.0, 135.0],  # right-eye outer corner (image left)
        [225.0, -170.0, 135.0],   # left-eye outer corner (image right)
        [-150.0, 150.0, 125.0],   # right mouth corner (image left)
        [150.0, 150.0, 125.0],    # left mouth corner (image right)
    ],
    dtype=np.float64,
)


def camera_matrix(width: int, height: int) -> np.ndarray:
    """Pinhole intrinsics approximated from the image size (f = width)."""
    f = float(width)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def rotation_to_euler(rmat: np.ndarray) -> Tuple[float, float, float]:
    """Decompose a rotation matrix into (pitch, yaw, roll) in degrees.

    Conventions (camera frame, subject facing the camera):

    * ``yaw``   > 0: the nose points towards image-right
    * ``pitch`` > 0: the nose points upwards
    * ``roll``  > 0: the head tilts clockwise as seen in the image
    """
    r = np.asarray(rmat, dtype=float)
    sy = math.hypot(r[0, 0], r[1, 0])
    if sy > 1e-6:
        rx = math.atan2(r[2, 1], r[2, 2])
        ry = math.atan2(-r[2, 0], sy)
        rz = math.atan2(r[1, 0], r[0, 0])
    else:  # gimbal lock
        rx = math.atan2(-r[1, 2], r[1, 1])
        ry = math.atan2(-r[2, 0], sy)
        rz = 0.0
    return -math.degrees(rx), -math.degrees(ry), math.degrees(rz)


def euler_to_rotation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Inverse of :func:`rotation_to_euler` (used by tests and simulations)."""
    rx, ry, rz = math.radians(-pitch), math.radians(-yaw), math.radians(roll)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


def estimate_head_pose(
    image_points: np.ndarray, width: int, height: int
) -> Optional[Tuple[float, float, float]]:
    """Estimate (pitch, yaw, roll) from six 2-D anchors via Perspective-n-Point.

    Returns ``None`` if the solver fails.
    """
    import cv2  # local import keeps the module importable without OpenCV

    pts = np.ascontiguousarray(np.asarray(image_points, dtype=np.float64).reshape(-1, 2))
    k = camera_matrix(width, height)

    # Seed the iterative solver with a frontal pose at a plausible depth.
    # Without this seed Levenberg-Marquardt occasionally converges to the
    # mirrored solution (roll close to 180 degrees) on real faces.
    eye_span_px = max(distance(pts[2], pts[3]), 1.0)
    model_eye_span = distance(HEAD_MODEL_3D[2], HEAD_MODEL_3D[3])
    depth = k[0, 0] * model_eye_span / eye_span_px
    tvec0 = np.array(
        [[(pts[0, 0] - k[0, 2]) * depth / k[0, 0]], [(pts[0, 1] - k[1, 2]) * depth / k[1, 1]], [depth]]
    )
    ok, rvec, _tvec = cv2.solvePnP(
        HEAD_MODEL_3D,
        pts,
        k,
        np.zeros((4, 1)),
        np.zeros((3, 1)),
        tvec0,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    return rotation_to_euler(rmat)


def wrap_degrees(angle: float) -> float:
    """Wrap an angle to the interval (-180, 180]."""
    a = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a
