"""Debounced behavioural events.

Raw per-frame conditions flicker (a single mis-detected frame should not
register as "user left the desk"). Each ``Rule`` therefore has:

* ``onset``  - how long the condition must hold before the event starts,
* ``offset`` - how long it must be false before the event ends.

When an event starts, its recorded start time is back-dated to the moment the
condition first became true, so durations reflect reality rather than the
debounce delay. Instant events (bookmarks, yawns, gesture commands) bypass the
rules and are logged directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"


@dataclass
class Event:
    kind: str
    start: float
    end: Optional[float] = None
    severity: str = SEVERITY_INFO
    message: str = ""
    data: Dict[str, object] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.end is None

    def duration(self, now: Optional[float] = None) -> float:
        end = self.end if self.end is not None else now
        return 0.0 if end is None else max(end - self.start, 0.0)

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["duration"] = round(self.duration(), 3)
        return d


@dataclass
class Rule:
    kind: str
    onset: float
    offset: float = 0.5
    severity: str = SEVERITY_WARNING
    message: str = ""
    _true_since: Optional[float] = None
    _false_since: Optional[float] = None
    _event: Optional[Event] = None


class EventEngine:
    def __init__(self, on_event: Optional[Callable[[Event], None]] = None) -> None:
        self._rules: Dict[str, Rule] = {}
        self.history: List[Event] = []
        self._on_event = on_event

    def add_rule(self, kind: str, onset: float, offset: float = 0.5,
                 severity: str = SEVERITY_WARNING, message: str = "") -> None:
        self._rules[kind] = Rule(kind=kind, onset=onset, offset=offset, severity=severity, message=message)

    def update(self, t: float, conditions: Dict[str, bool]) -> List[Event]:
        """Advance every rule. Returns events that *started* on this call."""
        started: List[Event] = []
        for kind, rule in self._rules.items():
            cond = bool(conditions.get(kind, False))
            if cond:
                rule._false_since = None
                if rule._true_since is None:
                    rule._true_since = t
                if rule._event is None and t - rule._true_since >= rule.onset:
                    rule._event = Event(kind, rule._true_since, severity=rule.severity, message=rule.message)
                    self.history.append(rule._event)
                    started.append(rule._event)
                    self._emit(rule._event)
            else:
                rule._true_since = None
                if rule._event is not None:
                    if rule._false_since is None:
                        rule._false_since = t
                    if t - rule._false_since >= rule.offset:
                        rule._event.end = rule._false_since
                        self._emit(rule._event)
                        rule._event = None
                        rule._false_since = None
        return started

    def instant(self, kind: str, t: float, message: str = "", severity: str = SEVERITY_INFO,
                data: Optional[Dict[str, object]] = None) -> Event:
        event = Event(kind, t, end=t, severity=severity, message=message, data=dict(data or {}))
        self.history.append(event)
        self._emit(event)
        return event

    def active(self) -> List[Event]:
        return [r._event for r in self._rules.values() if r._event is not None]

    def close_all(self, t: float) -> None:
        for rule in self._rules.values():
            if rule._event is not None:
                rule._event.end = t
                self._emit(rule._event)
                rule._event = None
            rule._true_since = rule._false_since = None

    def _emit(self, event: Event) -> None:
        if self._on_event is not None:
            self._on_event(event)

    def totals(self, now: float) -> Dict[str, Dict[str, float]]:
        """Count and cumulative duration per event kind."""
        out: Dict[str, Dict[str, float]] = {}
        for e in self.history:
            entry = out.setdefault(e.kind, {"count": 0, "seconds": 0.0})
            entry["count"] += 1
            entry["seconds"] += e.duration(now)
        return out
