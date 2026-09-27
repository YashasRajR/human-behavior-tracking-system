"""Visual design tokens for the dashboard.

A single dark palette with one accent colour and three semantic colours
(good / warning / critical) keeps the interface calm: colour is only used when
it carries meaning. All colours are BGR tuples, as expected by OpenCV.
"""

from __future__ import annotations

from typing import Tuple

import cv2

Color = Tuple[int, int, int]


def hex_to_bgr(value: str) -> Color:
    value = value.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


BG = hex_to_bgr("#0f1216")
SURFACE = hex_to_bgr("#171b21")
CARD = hex_to_bgr("#1c2129")
BORDER = hex_to_bgr("#2a313b")
TRACK = hex_to_bgr("#2b323c")
TEXT = hex_to_bgr("#e8ebf0")
MUTED = hex_to_bgr("#8a94a6")
SUBTLE = hex_to_bgr("#5d6675")

ACCENT = hex_to_bgr("#3b82f6")
GOOD = hex_to_bgr("#22a06b")
WARNING = hex_to_bgr("#e5a50a")
CRITICAL = hex_to_bgr("#e5484d")
RECORD = hex_to_bgr("#ef4444")

MESH = hex_to_bgr("#5eead4")
IRIS = hex_to_bgr("#f472b6")
HAND = hex_to_bgr("#fbbf24")
POSE = hex_to_bgr("#60a5fa")

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA

TOP_BAR_HEIGHT = 40
VIDEO_HEIGHT = 720
PANEL_PADDING = 14
CARD_GAP = 10


def lerp_color(a: Color, b: Color, t: float) -> Color:
    t = min(max(t, 0.0), 1.0)
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))  # type: ignore[return-value]


def score_color(score: float) -> Color:
    """Map a 0-100 score to critical -> warning -> good."""
    if score < 50:
        return lerp_color(CRITICAL, WARNING, score / 50.0)
    return lerp_color(WARNING, GOOD, (score - 50.0) / 50.0)


def severity_color(severity: str) -> Color:
    return {"critical": CRITICAL, "warning": WARNING}.get(severity, ACCENT)
