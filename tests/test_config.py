import pytest

from hbts.config import DEFAULTS, apply_overrides, deep_merge, load_config


def test_defaults_are_complete():
    cfg = load_config()
    assert cfg == DEFAULTS
    assert cfg is not DEFAULTS


def test_overrides_parse_types():
    cfg = apply_overrides(DEFAULTS, [
        "source.camera=2", "attention.yaw_tolerance=25.5", "modules.pose=false",
        "ui.window=Lab", "new.section.key=on",
    ])
    assert cfg["source"]["camera"] == 2
    assert cfg["attention"]["yaw_tolerance"] == 25.5
    assert cfg["modules"]["pose"] is False
    assert cfg["ui"]["window"] == "Lab"
    assert cfg["new"]["section"]["key"] is True
    assert DEFAULTS["source"]["camera"] == 0  # defaults untouched


def test_bad_override():
    with pytest.raises(ValueError):
        apply_overrides(DEFAULTS, ["no_equals_sign"])


def test_deep_merge_keeps_siblings():
    merged = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}})
    assert merged == {"a": {"x": 1, "y": 3}}


def test_yaml_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("attention:\n  weights:\n    head: 1.0\n")
    cfg = load_config(str(path), ["attention.weights.eyes=0"])
    assert cfg["attention"]["weights"] == {"head": 1.0, "gaze": 0.35, "eyes": 0}


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_repository_example_config_loads():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    cfg = load_config(str(example))
    assert cfg["commands"]["bindings"]["Thumb_Up"] == "bookmark"
