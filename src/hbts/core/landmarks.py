"""Landmark index tables for the MediaPipe face mesh (478 points) and pose model.

Naming follows the *subject's* anatomy: ``RIGHT_EYE`` is the subject's right
eye, which appears on the left side of an un-mirrored image.
"""

from __future__ import annotations

# --- Eyes -----------------------------------------------------------------
# Six-point contours in the order used by the Eye Aspect Ratio
# (Soukupova & Cech, 2016): p1 outer corner, p2/p3 upper lid, p4 inner corner,
# p5/p6 lower lid.
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR = (263, 387, 385, 362, 380, 373)

# Corners and lid mid-points used for the iris ratio.
RIGHT_EYE_CORNERS = (33, 133)     # outer, inner
LEFT_EYE_CORNERS = (263, 362)     # outer, inner
RIGHT_EYE_LIDS = (159, 145)       # upper, lower
LEFT_EYE_LIDS = (386, 374)        # upper, lower

# Iris: centre followed by four boundary points (requires the refined mesh).
RIGHT_IRIS = (468, 469, 470, 471, 472)
LEFT_IRIS = (473, 474, 475, 476, 477)

# Full eye outlines, used by the privacy renderer and the mesh overlay.
RIGHT_EYE_OUTLINE = (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7)
LEFT_EYE_OUTLINE = (263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249)

# --- Mouth ------------------------------------------------------------------
MOUTH_CORNERS = (61, 291)
# Three inner-lip vertical pairs (upper, lower) for the Mouth Aspect Ratio.
MOUTH_VERTICAL_PAIRS = ((81, 178), (13, 14), (311, 402))
LIPS_OUTLINE = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146)

# --- Face outline / pose anchors -------------------------------------------
FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
)

# Six anchors for Perspective-n-Point head pose, in the same order as
# ``geometry.HEAD_MODEL_3D``: nose tip, chin, right-eye outer corner,
# left-eye outer corner, right mouth corner, left mouth corner.
HEAD_POSE_ANCHORS = (1, 152, 33, 263, 61, 291)

NOSE_TIP = 1

# --- Pose (BlazePose, 33 points) -------------------------------------------
POSE_NOSE = 0
POSE_LEFT_EAR = 7
POSE_RIGHT_EAR = 8
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24

# Upper-body skeleton edges drawn by the dashboard.
POSE_UPPER_BODY_EDGES = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
)

# --- Hands (21 points) -----------------------------------------------------
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
