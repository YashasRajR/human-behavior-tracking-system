import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from hbts.analytics.engine import BehaviorEngine, BehaviorState  # noqa: E402
from hbts.core.types import FrameAnalysis  # noqa: E402
from hbts.ui import theme as T  # noqa: E402
from hbts.ui.dashboard import Dashboard, ViewStatus, attention_label  # noqa: E402
from hbts.vision.face import extract_face_features  # noqa: E402


def test_layout_dimensions():
    dash = Dashboard(panel_width=380)
    img = np.zeros((480, 640, 3), np.uint8)
    out = dash.render(img, None, BehaviorState(), ViewStatus())
    assert out.shape == (T.TOP_BAR_HEIGHT + T.VIDEO_HEIGHT, 960 + 380, 3)


def test_privacy_mode_hides_camera_image(frontal_landmarks, cfg):
    lm, w, h = frontal_landmarks
    face = extract_face_features(lm, w, h)
    frame = FrameAnalysis(t=1.0, index=30, width=w, height=h, face=face)
    state = BehaviorEngine(cfg).update(frame)
    status = ViewStatus(privacy=True)

    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    b = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    out_a = Dashboard().render(a, frame, state, status)
    out_b = Dashboard().render(b, frame, state, status)
    assert np.array_equal(out_a, out_b)  # nothing from the camera image leaks through


def test_all_overlays_render(frontal_landmarks, cfg):
    lm, w, h = frontal_landmarks
    face = extract_face_features(lm, w, h)
    engine = BehaviorEngine(cfg)
    dash = Dashboard()
    img = np.full((h, w, 3), 90, np.uint8)
    for i in range(10):
        frame = FrameAnalysis(t=i / 30, index=i, width=w, height=h, face=face if i % 2 else None)
        state = engine.update(frame)
        dash.notify("Test notification", "warning", state.t)
        for status in (ViewStatus(), ViewStatus(show_help=True), ViewStatus(paused=True, progress=0.4)):
            assert dash.render(img, frame, state, status).dtype == np.uint8


def test_attention_labels():
    assert attention_label(90, True)[0] == "Focused"
    assert attention_label(60, True)[0] == "Wandering"
    assert attention_label(10, True)[0] == "Distracted"
    assert attention_label(90, False)[0] == "Away"
    assert attention_label(90, True, calibrating=True)[0] == "Calibrating"
