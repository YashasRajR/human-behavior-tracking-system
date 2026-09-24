import pytest

from hbts.analytics.events import EventEngine


def drive(engine, kind, pattern, fps=10.0, t0=0.0):
    """pattern: list of (seconds, bool)."""
    t = t0
    for seconds, cond in pattern:
        for _ in range(int(round(seconds * fps))):
            engine.update(t, {kind: cond})
            t += 1.0 / fps
    return t


def test_onset_debounce_and_backdated_start():
    e = EventEngine()
    e.add_rule("distracted", onset=2.0, offset=0.5)
    drive(e, "distracted", [(1.0, False), (1.5, True), (1.0, False)])
    assert e.history == []  # 1.5 s < onset
    drive(e, "distracted", [(3.0, True), (1.0, False)], t0=10.0)
    assert len(e.history) == 1
    ev = e.history[0]
    assert ev.start == pytest.approx(10.0)
    assert ev.end == pytest.approx(13.0, abs=0.11)
    assert ev.duration() == pytest.approx(3.0, abs=0.11)


def test_offset_bridges_short_gaps():
    e = EventEngine()
    e.add_rule("absent", onset=0.5, offset=1.0)
    drive(e, "absent", [(2.0, True), (0.3, False), (2.0, True), (2.0, False)])
    assert len(e.history) == 1


def test_callbacks_and_active():
    seen = []
    e = EventEngine(on_event=lambda ev: seen.append((ev.kind, ev.end is None)))
    e.add_rule("x", onset=0.0, offset=0.0)
    e.update(0.0, {"x": True})
    assert [ev.kind for ev in e.active()] == ["x"]
    e.update(0.1, {"x": False})
    assert seen == [("x", True), ("x", False)]
    assert e.active() == []


def test_instant_and_totals():
    e = EventEngine()
    e.instant("bookmark", 5.0, "Bookmark")
    e.instant("bookmark", 9.0, "Bookmark")
    totals = e.totals(10.0)
    assert totals["bookmark"]["count"] == 2
    assert totals["bookmark"]["seconds"] == 0.0


def test_close_all_ends_open_events():
    e = EventEngine()
    e.add_rule("drowsy", onset=0.0)
    e.update(1.0, {"drowsy": True})
    e.close_all(4.0)
    assert e.history[0].end == 4.0
    assert e.active() == []
