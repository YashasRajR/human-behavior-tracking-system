"""Overlays drawn on the video area: landmarks, privacy view and modal layers."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from hbts.core import landmarks as L
from hbts.core.types import FaceObservation, HandObservation, PoseObservation
from hbts.ui import theme as T
from hbts.ui import widgets as W

COMMAND_LABELS = {
    "bookmark": "Bookmark",
    "snapshot": "Snapshot",
    "toggle_privacy": "Privacy mode",
    "toggle_recording": "Recording",
}


def _pts(lm: np.ndarray, idx: Sequence[int], scale: float) -> np.ndarray:
    return (lm[list(idx), :2] * scale).astype(np.int32)


def draw_face(img: np.ndarray, face: FaceObservation, scale: float, strong: bool = False) -> None:
    lm = face.landmarks
    color = T.MESH
    thickness = 2 if strong else 1
    for contour, closed in ((L.FACE_OVAL, True), (L.RIGHT_EYE_OUTLINE, True),
                            (L.LEFT_EYE_OUTLINE, True), (L.LIPS_OUTLINE, True)):
        cv2.polylines(img, [_pts(lm, contour, scale)], closed, color, thickness, T.AA)
    for iris in (L.RIGHT_IRIS, L.LEFT_IRIS):
        pts = lm[list(iris), :2] * scale
        center = pts[0]
        radius = float(np.mean(np.linalg.norm(pts[1:] - center, axis=1)))
        cv2.circle(img, tuple(center.astype(int)), max(int(radius), 2), T.IRIS, thickness, T.AA)
        cv2.circle(img, tuple(center.astype(int)), 2, T.IRIS, -1, T.AA)

    # Head direction vector from the nose tip
    nose = lm[L.NOSE_TIP, :2] * scale
    length = 0.35 * face.bbox[2] * scale
    dx = np.sin(np.radians(face.yaw)) * length
    dy = -np.sin(np.radians(face.pitch)) * length
    tip = (int(nose[0] + dx), int(nose[1] + dy))
    cv2.arrowedLine(img, tuple(nose.astype(int)), tip, T.ACCENT, 2, T.AA, tipLength=0.2)


def draw_hands(img: np.ndarray, hands: Iterable[HandObservation], scale: float, labels: bool = True) -> None:
    for hand in hands:
        pts = (hand.landmarks[:, :2] * scale).astype(int)
        for a, b in L.HAND_EDGES:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), T.HAND, 2, T.AA)
        for p in pts:
            cv2.circle(img, tuple(p), 3, T.TEXT, -1, T.AA)
        if labels:
            x, y, _, _ = hand.bbox
            caption = hand.handedness if hand.gesture == "None" else f"{hand.handedness}  {hand.gesture.replace('_', ' ')}"
            W.pill(img, caption, int(x * scale), max(int(y * scale) - 30, 4), T.BG, T.HAND, 0.42)


def draw_pose(img: np.ndarray, pose: PoseObservation, scale: float) -> None:
    pts = (pose.landmarks[:, :2] * scale).astype(int)
    vis = pose.visibility
    for a, b in L.POSE_UPPER_BODY_EDGES:
        if vis[a] > 0.5 and vis[b] > 0.5:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), T.POSE, 2, T.AA)
    for i in (11, 12, 13, 14, 15, 16, 23, 24):
        if vis[i] > 0.5:
            cv2.circle(img, tuple(pts[i]), 4, T.POSE, -1, T.AA)
    # Neck: shoulder midpoint to ear midpoint
    if min(vis[7], vis[8], vis[11], vis[12]) > 0.5:
        neck_base = ((pts[11] + pts[12]) // 2)
        head = ((pts[7] + pts[8]) // 2)
        cv2.line(img, tuple(neck_base), tuple(head), T.POSE, 2, T.AA)
        cv2.circle(img, tuple(neck_base), 4, T.POSE, -1, T.AA)


def privacy_canvas(shape: Tuple[int, int, int]) -> np.ndarray:
    """Neutral background used instead of the camera image in privacy mode."""
    h, w = shape[:2]
    canvas = np.empty(shape, dtype=np.uint8)
    top, bottom = np.array(T.SURFACE, float), np.array(T.BG, float)
    grad = np.linspace(0, 1, h)[:, None] * (bottom - top) + top
    canvas[:] = grad[:, None, :].astype(np.uint8)
    step = 40
    for x in range(0, w, step):
        cv2.line(canvas, (x, 0), (x, h), T.SURFACE, 1)
    for y in range(0, h, step):
        cv2.line(canvas, (0, y), (w, y), T.SURFACE, 1)
    return canvas


def command_ring(img: np.ndarray, anchor: Tuple[int, int], command: str, progress: float,
                 scale: float) -> None:
    c = (int(anchor[0] * scale), int(anchor[1] * scale))
    color = T.GOOD if progress >= 1.0 else T.ACCENT
    cv2.circle(img, c, 34, T.BG, -1, T.AA)
    W.ring(img, c, 30, progress, color, 5)
    name = COMMAND_LABELS.get(command, command)
    W.pill(img, name, c[0] - W.text_size(name, 0.42)[0] // 2 - 10, c[1] + 42, T.TEXT, T.BG, 0.42)


def calibration_overlay(img: np.ndarray, progress: float, face_visible: bool) -> None:
    h, w = img.shape[:2]
    W.blend_rect(img, 0, 0, w, h, T.BG, 0.55)
    c = (w // 2, h // 2 - 20)
    W.ring(img, c, 54, progress, T.ACCENT, 6)
    W.text(img, f"{int(progress * 100)}%", (c[0], c[1] + 10), 0.9, T.TEXT, 2, T.FONT_BOLD, "center")
    W.text(img, "Calibrating", (c[0], c[1] + 100), 0.8, T.TEXT, 2, T.FONT_BOLD, "center")
    hint = ("Sit naturally and look at the screen" if face_visible
            else "Waiting for a face - move into the camera view")
    W.text(img, hint, (c[0], c[1] + 130), 0.55, T.MUTED, 1, align="center")


def alert_banner(img: np.ndarray, message: str, severity: str, t: float) -> None:
    w = img.shape[1]
    color = T.severity_color(severity)
    pulse = 0.55 + 0.25 * (0.5 + 0.5 * np.sin(t * 6.0))
    tw, _ = W.text_size(message, 0.6, 2, T.FONT_BOLD)
    bw = tw + 60
    x = (w - bw) // 2
    W.blend_rect(img, x, 16, bw, 44, color, pulse)
    W.rounded_rect(img, x, 16, bw, 44, 8, color, 2)
    W.text(img, message, (w // 2, 45), 0.6, T.TEXT, 2, T.FONT_BOLD, "center")


def toasts(img: np.ndarray, items: List[Tuple[str, str, float]], now: float, lifetime: float = 4.0) -> None:
    """Stack of recent notifications at the bottom-left of the video."""
    h = img.shape[0]
    y = h - 20
    for message, severity, born in reversed(items[-4:]):
        age = now - born
        if age > lifetime:
            continue
        alpha = 0.85 if age < lifetime - 0.6 else 0.85 * max(lifetime - age, 0.0) / 0.6
        tw, _ = W.text_size(message, 0.5, 1)
        bw, bh = tw + 34, 34
        y -= bh
        W.blend_rect(img, 16, y, bw, bh, T.SURFACE, alpha)
        cv2.rectangle(img, (16, y), (19, y + bh), T.severity_color(severity), -1)
        W.text(img, message, (30, y + 22), 0.5, T.TEXT, 1)
        y -= 8


HELP_ROWS = (
    ("H", "Show or hide this help"),
    ("C", "Recalibrate baselines"),
    ("M", "Toggle landmark overlay"),
    ("P", "Toggle privacy mode"),
    ("R", "Pause or resume recording"),
    ("B", "Add a bookmark"),
    ("S", "Save a snapshot"),
    ("Space", "Pause or resume video"),
    ("Q / Esc", "Quit and write the session report"),
)

GESTURE_ROWS = (
    ("Thumb up", "Bookmark"),
    ("Victory", "Snapshot"),
    ("Open palm", "Privacy mode"),
    ("Closed fist", "Recording on / off"),
)


def help_overlay(img: np.ndarray, hold_seconds: float) -> None:
    h, w = img.shape[:2]
    W.blend_rect(img, 0, 0, w, h, T.BG, 0.78)
    bw, bh = min(620, w - 40), 470
    x, y = (w - bw) // 2, (h - bh) // 2
    W.rounded_rect(img, x, y, bw, bh, 12, T.CARD)
    W.rounded_rect(img, x, y, bw, bh, 12, T.BORDER, 1)
    W.text(img, "Controls", (x + 24, y + 40), 0.75, T.TEXT, 2, T.FONT_BOLD)
    W.label(img, "Keyboard", (x + 24, y + 72))
    ry = y + 84
    for key, desc in HELP_ROWS:
        W.keycap(img, key, x + 24, ry)
        W.text(img, desc, (x + 124, ry + 17), 0.5, T.TEXT, 1)
        ry += 30
    W.label(img, f"Gestures (hold {hold_seconds:.1f} s)", (x + 24, ry + 18))
    ry += 30
    col = x + 24
    for i, (g, desc) in enumerate(GESTURE_ROWS):
        cx = col + (i % 2) * (bw // 2 - 12)
        cy = ry + (i // 2) * 26
        W.text(img, g, (cx, cy + 14), 0.5, T.HAND, 1)
        W.text(img, desc, (cx + 110, cy + 14), 0.5, T.MUTED, 1)


def no_face_hint(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    W.pill(img, "No face detected", w // 2 - 80, h - 60, T.TEXT, T.SURFACE, 0.5, dot=T.WARNING)


def paused_overlay(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    W.blend_rect(img, 0, 0, w, h, T.BG, 0.35)
    cx, cy = w // 2, h // 2
    cv2.rectangle(img, (cx - 26, cy - 34), (cx - 8, cy + 34), T.TEXT, -1)
    cv2.rectangle(img, (cx + 8, cy - 34), (cx + 26, cy + 34), T.TEXT, -1)


def progress_strip(img: np.ndarray, fraction: Optional[float]) -> None:
    if fraction is None:
        return
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 3), (w, h), T.TRACK, -1)
    cv2.rectangle(img, (0, h - 3), (int(w * fraction), h), T.ACCENT, -1)
