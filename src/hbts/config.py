"""Layered configuration.

Values are resolved in increasing order of precedence:

1. ``DEFAULTS`` defined below (always complete)
2. an optional YAML file (``--config my.yaml``)
3. dotted command-line overrides (``--set attention.weights.gaze=0.5``)

Every consumer receives a plain nested ``dict``; keeping configuration as data
rather than as scattered constants makes experiments reproducible, because the
resolved configuration is written into every session folder.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

DEFAULTS: Dict[str, Any] = {
    "source": {
        "camera": 0,
        "width": 1280,
        "height": 720,
        "mirror": True,
    },
    "modules": {
        "face": True,
        "hands": True,
        "pose": True,
    },
    "models": {
        "dir": "models",
        "face": "face_landmarker.task",
        "hands": "gesture_recognizer.task",
        "pose": "pose_landmarker_lite.task",
    },
    "smoothing": {
        "min_cutoff": 1.2,
        "beta": 0.03,
    },
    "blink": {
        "close_ratio": 0.72,        # eye counts as closed below ratio * open-eye baseline
        "open_ratio": 0.82,         # and re-opens above this ratio (hysteresis)
        "default_ear": 0.28,        # baseline used until calibration finishes
        "min_duration": 0.05,       # seconds; shorter dips are landmark noise
        "max_duration": 0.50,       # seconds; longer closures are not blinks
        "rate_window": 60.0,        # seconds for the blinks-per-minute estimate
    },
    "fatigue": {
        "perclos_window": 60.0,     # seconds
        "perclos_threshold": 0.15,  # fraction of time eyes are closed
        "microsleep_seconds": 1.0,
        "yawn_mar": 0.55,
        "yawn_seconds": 1.2,
    },
    "calibration": {
        "seconds": 4.0,
        "min_samples": 30,
    },
    "attention": {
        "yaw_tolerance": 22.0,      # degrees of head yaw before the score halves
        "pitch_tolerance": 18.0,
        "gaze_tolerance": 0.16,     # iris ratio deviation before the score halves
        "time_constant": 0.8,       # seconds, output smoothing
        "weights": {"head": 0.45, "gaze": 0.35, "eyes": 0.20},
    },
    "posture": {
        "neck_angle_limit": 18.0,   # degrees beyond calibrated neutral
        "head_drop_ratio": 0.80,    # fraction of calibrated head height
        "shoulder_tilt_limit": 9.0,
    },
    "events": {
        "distracted_seconds": 3.0,
        "absent_seconds": 2.0,
        "slouch_seconds": 8.0,
        "low_attention": 45.0,
    },
    "commands": {
        "enabled": True,
        "hold_seconds": 1.2,
        "cooldown_seconds": 2.0,
        "min_score": 0.6,
        "bindings": {
            "Thumb_Up": "bookmark",
            "Victory": "snapshot",
            "Open_Palm": "toggle_privacy",
            "Closed_Fist": "toggle_recording",
        },
    },
    "recording": {
        "enabled": True,
        "dir": "sessions",
        "sample_hz": 10.0,
    },
    "ui": {
        "window": "HBTS - Human Behavior Tracking",
        "panel_width": 380,
        "show_mesh": True,
        "privacy": False,
        "fullscreen": False,
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_scalar(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply ``key.sub=value`` strings to a configuration dictionary."""
    result = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like key.path=value")
        path, raw = item.split("=", 1)
        keys = [k for k in path.strip().split(".") if k]
        if not keys:
            raise ValueError(f"override '{item}' has an empty key")
        node = result
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"'{key}' in '{path}' is not a section")
        node[keys[-1]] = _parse_scalar(raw)
    return result


def load_config(path: Optional[str] = None, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    """Resolve the effective configuration."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        import yaml

        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"configuration file not found: {file_path}")
        with file_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError("the configuration file must contain a mapping at the top level")
        cfg = deep_merge(cfg, data)
    return apply_overrides(cfg, overrides)
