import json
from pathlib import Path

import numpy as np
import pytest

from hbts.config import load_config

DATA = Path(__file__).parent / "data"


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def frontal_landmarks():
    """(landmarks_px, width, height) for a recorded near-frontal face."""
    data = json.loads((DATA / "face_landmarks_frontal.json").read_text())
    w, h = data["width"], data["height"]
    lm = np.array(data["landmarks"], dtype=np.float32) * np.array([w, h, w], dtype=np.float32)
    return lm, w, h

