import json

from hbts.io.report import compute_stats, generate_report, gesture_onsets, load_session


def make_session(tmp_path, blink_rate=6.0):
    session = tmp_path / "20260101-120000"
    session.mkdir()
    rows = []
    for i in range(1200):  # 2 minutes at 10 Hz
        t = i / 10.0
        rows.append({
            "t": t, "face": not (60 <= t < 65), "calibrating": t < 4,
            "attention": 90.0 if t < 60 else 40.0, "on_screen": t < 60,
            "ear": 0.3, "eyes_closed": False, "blink_count": int(t / 10), "blink_rate": blink_rate,
            "perclos": 0.05, "mar": 0.1, "yawn_count": 0, "fatigue": "Alert",
            "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "gaze_h": 0.5, "gaze_v": 0.5,
            "head": "Center", "gaze": "Center", "posture": "Upright",
            "gestures": ["Thumb_Up"] if 20 <= t < 22 or 30 <= t < 31 else [],
        })
    (session / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n{truncated")
    events = [
        {"kind": "distracted", "start": 70.0, "end": 82.0, "severity": "warning",
         "message": "Looking away from screen", "duration": 12.0},
        {"kind": "bookmark", "start": 90.0, "end": 90.0, "severity": "info", "message": "Bookmark", "duration": 0.0},
    ]
    (session / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    (session / "summary.json").write_text(json.dumps({"started_at": "2026-01-01T12:00:00"}))
    return session


def test_gesture_onsets_counts_rising_edges():
    rows = [{"gestures": g} for g in ([], ["A"], ["A"], [], ["A", "B"], ["B"])]
    counts = gesture_onsets(rows)
    assert counts["A"] == 2 and counts["B"] == 1


def test_stats_and_insights(tmp_path):
    data = load_session(make_session(tmp_path))
    stats = compute_stats(data)
    assert stats["distractions"] == 1
    assert stats["longest_distraction"] == 12.0
    assert stats["gestures"]["Thumb_Up"] == 2
    assert stats["calibration_seconds"] > 3.5
    joined = " ".join(stats["insights"])
    assert "below the typical resting range" in joined
    assert "minute 1" in joined and "minute 2" in joined


def test_generate_report(tmp_path):
    session = make_session(tmp_path)
    out = generate_report(str(session))
    html = out.read_text()
    assert out.name == "report.html"
    for heading in ("Attention over time", "Eye metrics", "Head orientation", "Event log"):
        assert heading in html
    assert "<svg" in html and "Looking away from screen" in html
    assert "01 Jan 2026" in html
    assert "http" not in html  # fully self-contained
