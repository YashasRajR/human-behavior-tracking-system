import csv
import json

from hbts.io.recorder import SessionRecorder


def sample(t, **extra):
    row = {"t": t, "face": True, "attention": 80.0, "gestures": ["Thumb_Up"]}
    row.update(extra)
    return row


def test_writes_session_files(tmp_path):
    rec = SessionRecorder(str(tmp_path), sample_hz=10.0, config={"a": 1}, name="s1")
    for i in range(300):  # 10 s at 30 FPS
        rec.record(sample(i / 30.0))
    rec.event({"kind": "bookmark", "start": 1.0, "end": 1.0})
    path = rec.close({"blinks": 3})

    metrics = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    assert 95 <= len(metrics) <= 101  # 10 Hz for 10 s
    with (path / "metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(metrics)
    assert rows[0]["gestures"] == "Thumb_Up"

    events = (path / "events.jsonl").read_text().splitlines()
    assert json.loads(events[0])["kind"] == "bookmark"
    summary = json.loads((path / "summary.json").read_text())
    assert summary["blinks"] == 3 and summary["samples"] == len(metrics)
    assert json.loads((path / "config.json").read_text()) == {"a": 1}


def test_sampling_rate_with_irregular_frames(tmp_path):
    rec = SessionRecorder(str(tmp_path), sample_hz=10.0, name="s2")
    t = 0.0
    for _ in range(1000):  # 15 FPS, not a multiple of 10 Hz
        rec.record(sample(t))
        t += 1 / 15
    rec.close()
    n = len((tmp_path / "s2" / "metrics.jsonl").read_text().splitlines())
    assert abs(n / t - 10.0) < 0.5


def test_pause_skips_samples(tmp_path):
    rec = SessionRecorder(str(tmp_path), sample_hz=100.0, name="s3")
    rec.paused = True
    assert not rec.record(sample(0.0))
    rec.paused = False
    assert rec.record(sample(0.1))
    rec.close()


def test_unique_folder_names(tmp_path):
    a = SessionRecorder(str(tmp_path), name="same")
    b = SessionRecorder(str(tmp_path), name="same")
    assert a.path != b.path
    a.close()
    b.close()
