"""Drawing primitives for the OpenCV dashboard.

OpenCV has no widget toolkit, so the dashboard is composed from a small set of
primitives: translucent fills, rounded rectangles, anti-aliased text, pills,
progress bars, sparklines and progress rings.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from hbts.ui import theme as T
from hbts.ui.theme import Color


def blend_rect(img: np.ndarray, x: int, y: int, w: int, h: int, color: Color, alpha: float) -> None:
    """Fill a rectangle with ``color`` at the given opacity (in place)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    overlay = np.empty_like(roi)
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def rounded_rect(img: np.ndarray, x: int, y: int, w: int, h: int, r: int, color: Color,
                 thickness: int = -1) -> None:
    r = int(max(0, min(r, w // 2, h // 2)))
    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
            cv2.circle(img, (cx, cy), r, color, -1, T.AA)
        return
    cv2.line(img, (x + r, y), (x + w - r, y), color, thickness, T.AA)
    cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness, T.AA)
    cv2.line(img, (x, y + r), (x, y + h - r), color, thickness, T.AA)
    cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness, T.AA)
    for (cx, cy), start in (((x + r, y + r), 180), ((x + w - r, y + r), 270),
                            ((x + r, y + h - r), 90), ((x + w - r, y + h - r), 0)):
        cv2.ellipse(img, (cx, cy), (r, r), 0, start, start + 90, color, thickness, T.AA)


def text_size(s: str, scale: float = 0.5, thickness: int = 1, font: int = T.FONT) -> Tuple[int, int]:
    (w, h), _ = cv2.getTextSize(s, font, scale, thickness)
    return w, h


def text(img: np.ndarray, s: str, org: Tuple[int, int], scale: float = 0.5, color: Color = T.TEXT,
         thickness: int = 1, font: int = T.FONT, align: str = "left") -> int:
    """Draw text with its baseline at ``org``. Returns the rendered width."""
    w, _ = text_size(s, scale, thickness, font)
    x, y = org
    if align == "right":
        x -= w
    elif align == "center":
        x -= w // 2
    cv2.putText(img, s, (int(x), int(y)), font, scale, color, thickness, T.AA)
    return w


def label(img: np.ndarray, s: str, org: Tuple[int, int], color: Color = T.MUTED) -> int:
    """Small upper-case caption used for card titles and field names."""
    return text(img, s.upper(), org, 0.38, color, 1)


def pill(img: np.ndarray, s: str, x: int, y: int, fg: Color, bg: Color, scale: float = 0.42,
         dot: Optional[Color] = None, align: str = "left") -> int:
    """Rounded status badge. ``y`` is the top edge. Returns the badge width."""
    tw, th = text_size(s, scale, 1)
    pad_x, h = 10, th + 12
    extra = 14 if dot is not None else 0
    w = tw + 2 * pad_x + extra
    if align == "right":
        x -= w
    rounded_rect(img, x, y, w, h, h // 2, bg)
    if dot is not None:
        cv2.circle(img, (x + pad_x + 3, y + h // 2), 4, dot, -1, T.AA)
    text(img, s, (x + pad_x + extra, y + h - 6), scale, fg, 1)
    return w


def hbar(img: np.ndarray, x: int, y: int, w: int, h: int, fraction: float, color: Color,
         track: Color = T.TRACK) -> None:
    rounded_rect(img, x, y, w, h, h // 2, track)
    fw = int(round(w * min(max(fraction, 0.0), 1.0)))
    if fw >= h:
        rounded_rect(img, x, y, fw, h, h // 2, color)
    elif fw > 0:
        cv2.circle(img, (x + h // 2, y + h // 2), h // 2, color, -1, T.AA)


def centered_bar(img: np.ndarray, x: int, y: int, w: int, h: int, value: float, limit: float,
                 color: Color) -> None:
    """Bar that grows left or right from the centre; used for signed angles."""
    rounded_rect(img, x, y, w, h, h // 2, T.TRACK)
    mid = x + w // 2
    frac = max(-1.0, min(1.0, value / limit if limit else 0.0))
    end = int(mid + frac * (w // 2))
    x0, x1 = sorted((mid, end))
    if x1 - x0 >= 1:
        cv2.rectangle(img, (x0, y), (x1, y + h), color, -1)
    cv2.line(img, (mid, y - 2), (mid, y + h + 2), T.SUBTLE, 1, T.AA)


def sparkline(img: np.ndarray, x: int, y: int, w: int, h: int, values: Sequence[float],
              lo: float, hi: float, color: Color, fill_alpha: float = 0.18) -> None:
    if len(values) < 2 or hi <= lo:
        return
    n = len(values)
    xs = x + np.arange(n) * (w - 1) / (n - 1)
    v = np.clip((np.asarray(values, dtype=float) - lo) / (hi - lo), 0.0, 1.0)
    ys = y + (1.0 - v) * (h - 1)
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    if fill_alpha > 0:
        poly = np.vstack([pts, [[x + w - 1, y + h - 1], [x, y + h - 1]]]).astype(np.int32)
        layer = img[y:y + h, x:x + w].copy()
        shifted = poly - np.array([x, y])
        cv2.fillPoly(layer, [shifted], color, T.AA)
        cv2.addWeighted(layer, fill_alpha, img[y:y + h, x:x + w], 1.0 - fill_alpha, 0,
                        dst=img[y:y + h, x:x + w])
    cv2.polylines(img, [pts], False, color, 2, T.AA)


def ring(img: np.ndarray, center: Tuple[int, int], radius: int, progress: float, color: Color,
         thickness: int = 5, track: Optional[Color] = T.TRACK) -> None:
    """Circular progress indicator starting at 12 o'clock, clockwise."""
    if track is not None:
        cv2.circle(img, center, radius, track, thickness, T.AA)
    sweep = 360.0 * min(max(progress, 0.0), 1.0)
    if sweep > 0.5:
        cv2.ellipse(img, center, (radius, radius), -90, 0, sweep, color, thickness, T.AA)


def card(img: np.ndarray, x: int, y: int, w: int, h: int, title: str,
         accent: Optional[Color] = None) -> None:
    rounded_rect(img, x, y, w, h, 10, T.CARD)
    rounded_rect(img, x, y, w, h, 10, T.BORDER, 1)
    if accent is not None:
        cv2.line(img, (x + 1, y + 12), (x + 1, y + h - 12), accent, 2, T.AA)
    label(img, title, (x + 14, y + 22))


def keycap(img: np.ndarray, key: str, x: int, y: int) -> int:
    """Keyboard key glyph, top-left at (x, y). Returns its width."""
    tw, _ = text_size(key, 0.45, 1)
    w = max(tw + 14, 26)
    rounded_rect(img, x, y, w, 24, 5, T.SURFACE)
    rounded_rect(img, x, y, w, 24, 5, T.BORDER, 1)
    text(img, key, (x + w // 2, y + 17), 0.45, T.TEXT, 1, align="center")
    return w

