"""Self-contained HTML report for a recorded session.

The report has no external dependencies (no JavaScript libraries, no CDN, no
plotting package): charts are rendered as inline SVG, so the single HTML file
can be e-mailed or archived and will open identically anywhere.

Usage::

    hbts report sessions/20260930-142501
"""

from __future__ import annotations

import html
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PALETTE = {
    "attention": "#2f7de1",
    "blink": "#1a9e77",
    "perclos": "#d95f02",
    "yaw": "#7570b3",
    "pitch": "#e7298a",
    "band_warning": "rgba(217, 95, 2, 0.14)",
    "band_critical": "rgba(200, 30, 45, 0.18)",
    "band_calibration": "rgba(120, 130, 150, 0.16)",
}

BAND_EVENTS = {"distracted": "band_warning", "absent": "band_warning",
               "drowsy": "band_critical", "microsleep": "band_critical"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a truncated final line after a crash
    return rows


def load_session(session_dir: Path) -> Dict[str, Any]:
    session_dir = Path(session_dir)
    summary_path = session_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    return {
        "name": session_dir.name,
        "metrics": _read_jsonl(session_dir / "metrics.jsonl"),
        "events": _read_jsonl(session_dir / "events.jsonl"),
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def gesture_onsets(metrics: List[Dict[str, Any]]) -> Counter:
    """Count how many times each gesture *appeared* (rising edges)."""
    counts: Counter = Counter()
    previous: set = set()
    for row in metrics:
        current = set(row.get("gestures") or [])
        for g in current - previous:
            counts[g] += 1
        previous = current
    return counts


def _analysis_rows(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Samples recorded after calibration finished (calibration frames are excluded)."""
    return [r for r in metrics if not r.get("calibrating", False)]


def calibration_spans(metrics: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    spans: List[Tuple[float, float]] = []
    start: Optional[float] = None
    for r in metrics:
        if r.get("calibrating") and start is None:
            start = r["t"]
        elif not r.get("calibrating") and start is not None:
            spans.append((start, r["t"]))
            start = None
    if start is not None and metrics:
        spans.append((start, metrics[-1]["t"]))
    return spans


def compute_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    all_rows = data["metrics"]
    m = _analysis_rows(all_rows) or all_rows
    events = data["events"]
    present = [r for r in m if r.get("face")]
    duration = (all_rows[-1]["t"] - all_rows[0]["t"]) if len(all_rows) > 1 else 0.0

    distractions = [e for e in events if e.get("kind") == "distracted"]
    posture = Counter(r.get("posture", "Unknown") for r in m)
    stats = {
        "duration": duration,
        "samples": len(m),
        "presence": len(present) / len(m) if m else 0.0,
        "attention_mean": _mean([r["attention"] for r in present]),
        "focus_ratio": _mean([1.0 if r.get("on_screen") else 0.0 for r in present]),
        "blinks": m[-1].get("blink_count", 0) if m else 0,
        "blink_rate_mean": _mean([r["blink_rate"] for r in present if r["t"] - m[0]["t"] > 20]) if m else 0.0,
        "perclos_peak": max([r.get("perclos", 0.0) for r in present if r.get("perclos_ready", True)], default=0.0),
        "yawns": m[-1].get("yawn_count", 0) if m else 0,
        "distractions": len(distractions),
        "longest_distraction": max([e.get("duration", 0.0) for e in distractions], default=0.0),
        "posture": posture,
        "gestures": gesture_onsets(m),
        "event_counts": Counter(e.get("kind") for e in events),
    }
    stats["insights"] = insights(stats, m)
    stats["calibration_seconds"] = sum(b - a for a, b in calibration_spans(all_rows))
    return stats


def insights(stats: Dict[str, Any], metrics: List[Dict[str, Any]]) -> List[str]:
    """Plain-language observations generated from the statistics."""
    notes: List[str] = []
    if not metrics:
        return ["No samples were recorded in this session."]

    if stats["presence"] < 0.8:
        notes.append(f"A face was visible for only {stats['presence'] * 100:.0f}% of the session; "
                     "metrics below describe the visible portion.")

    att = stats["attention_mean"]
    level = "high" if att >= 75 else "moderate" if att >= 50 else "low"
    notes.append(f"Average attention was {att:.0f}/100 ({level}); eyes were on screen "
                 f"{stats['focus_ratio'] * 100:.0f}% of the time the face was visible.")

    # Best and worst minute
    by_minute: Dict[int, List[float]] = {}
    t0 = metrics[0]["t"]
    for r in metrics:
        if r.get("face"):
            by_minute.setdefault(int((r["t"] - t0) // 60), []).append(r["attention"])
    if len(by_minute) >= 2:
        means = {k: _mean(v) for k, v in by_minute.items()}
        best = max(means, key=means.get)
        worst = min(means, key=means.get)
        notes.append(f"Attention peaked in minute {best + 1} ({means[best]:.0f}) and was lowest in "
                     f"minute {worst + 1} ({means[worst]:.0f}).")

    if stats["distractions"]:
        notes.append(f"{stats['distractions']} distraction episode(s) were detected; the longest lasted "
                     f"{stats['longest_distraction']:.1f} s.")

    rate = stats["blink_rate_mean"]
    if rate > 0:
        if rate < 10:
            notes.append(f"Mean blink rate was {rate:.1f}/min, below the typical resting range of "
                         "roughly 12-20/min. Reduced blinking is common during focused screen work; "
                         "periodic breaks away from the screen may help.")
        elif rate > 30:
            notes.append(f"Mean blink rate was {rate:.1f}/min, above the typical resting range.")
        else:
            notes.append(f"Mean blink rate was {rate:.1f}/min, within the typical range.")

    if stats["perclos_peak"] >= 0.15:
        notes.append(f"PERCLOS reached {stats['perclos_peak'] * 100:.0f}%, which indicates periods of "
                     "reduced alertness.")
    if stats["yawns"]:
        notes.append(f"{stats['yawns']} yawn(s) were detected.")

    posture = stats["posture"]
    total = sum(posture.values())
    bad = sum(v for k, v in posture.items() if k in ("Slouching", "Leaning", "Uneven"))
    if total and bad / total > 0.2:
        notes.append(f"Posture was flagged for {bad / total * 100:.0f}% of the session.")
    return notes


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------

def _fmt_datetime(value: Any) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(str(value)).strftime("%d %b %Y, %H:%M")
    except ValueError:
        return str(value)


def _fmt_time(seconds: float) -> str:
    seconds = max(int(round(seconds)), 0)
    return f"{seconds // 60}:{seconds % 60:02d}"


def svg_line_chart(
    series: List[Tuple[str, List[float], List[float], str]],
    y_range: Tuple[float, float],
    bands: Optional[List[Tuple[float, float, str]]] = None,
    height: int = 220,
    width: int = 960,
    y_label: str = "",
    y_format: str = "{:.0f}",
    x_range: Optional[Tuple[float, float]] = None,
) -> str:
    """Render one or more line series sharing an x (time) axis as inline SVG."""
    pad_l, pad_r, pad_t, pad_b = 48, 16, 14, 30
    xs_all = [x for _, xs, _, _ in series for x in xs]
    if not xs_all:
        return '<p class="muted">No data.</p>'
    x_min, x_max = x_range if x_range else (min(xs_all), max(xs_all))
    if x_max - x_min < 1e-6:
        x_max = x_min + 1.0
    y_min, y_max = y_range
    if y_max - y_min < 1e-9:
        y_max = y_min + 1.0
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    def sx(x: float) -> float:
        return pad_l + (x - x_min) / (x_max - x_min) * pw

    def sy(y: float) -> float:
        y = min(max(y, y_min), y_max)
        return pad_t + (1.0 - (y - y_min) / (y_max - y_min)) * ph

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" preserveAspectRatio="none" class="chart">']
    for x0, x1, color_key in bands or []:
        parts.append(f'<rect x="{sx(x0):.1f}" y="{pad_t}" width="{max(sx(x1) - sx(x0), 1.5):.1f}" '
                     f'height="{ph}" fill="{PALETTE.get(color_key, color_key)}"/>')
    for i in range(5):
        yv = y_min + (y_max - y_min) * i / 4
        yy = sy(yv)
        parts.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{yy + 4:.1f}" text-anchor="end" class="tick">'
                     f'{html.escape(y_format.format(yv))}</text>')
    for i in range(6):
        xv = x_min + (x_max - x_min) * i / 5
        parts.append(f'<text x="{sx(xv):.1f}" y="{height - 8}" text-anchor="middle" class="tick">'
                     f'{_fmt_time(xv - x_min)}</text>')
    if y_label:
        parts.append(f'<text x="12" y="{pad_t + ph / 2:.1f}" class="tick" '
                     f'transform="rotate(-90 12 {pad_t + ph / 2:.1f})" text-anchor="middle">'
                     f'{html.escape(y_label)}</text>')
    for name, xs, ys, color in series:
        if not xs:
            continue
        step = max(1, len(xs) // 1500)  # keep the SVG light for long sessions
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in list(zip(xs, ys))[::step])
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8" '
                     f'stroke-linejoin="round"><title>{html.escape(name)}</title></polyline>')
    parts.append("</svg>")
    return "".join(parts)


def svg_bar_list(items: List[Tuple[str, float]], color: str, value_format: str = "{:.0f}") -> str:
    if not items:
        return '<p class="muted">None recorded.</p>'
    peak = max(v for _, v in items) or 1.0
    rows = []
    for label, value in items:
        pct = 100.0 * value / peak
        rows.append(
            f'<div class="bar-row"><span class="bar-label">{html.escape(str(label))}</span>'
            f'<span class="bar-track"><span class="bar-fill" style="width:{pct:.1f}%;background:{color}"></span></span>'
            f'<span class="bar-value">{html.escape(value_format.format(value))}</span></div>'
        )
    return "".join(rows)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

CSS = """
:root { --bg:#f6f7f9; --card:#ffffff; --text:#1d2330; --muted:#667085; --line:#e4e7ec; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1218; --card:#171b23; --text:#e6e9ef; --muted:#98a2b3; --line:#2a303b; }
}
* { box-sizing:border-box; }
body { margin:0; font:15px/1.5 "Segoe UI", system-ui, -apple-system, sans-serif; background:var(--bg); color:var(--text); }
main { max-width:1040px; margin:0 auto; padding:32px 20px 48px; }
header h1 { margin:0; font-size:26px; font-weight:650; letter-spacing:-0.01em; }
header p { margin:4px 0 0; color:var(--muted); }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:24px 0; }
.kpi { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.kpi .v { font-size:26px; font-weight:650; font-variant-numeric:tabular-nums; }
.kpi .l { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; }
section { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin-top:16px; }
section h2 { margin:0 0 12px; font-size:16px; font-weight:600; }
.chart { width:100%; height:220px; display:block; }
.grid { stroke:var(--line); stroke-width:1; }
.tick { fill:var(--muted); font-size:11px; }
.legend { display:flex; gap:16px; font-size:13px; color:var(--muted); margin-top:6px; flex-wrap:wrap; }
.legend i { display:inline-block; width:12px; height:3px; margin-right:6px; vertical-align:middle; }
.two { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.two section { margin-top:16px; }
@media (max-width:760px) { .two { grid-template-columns:1fr; } }
.bar-row { display:grid; grid-template-columns:110px 1fr 56px; gap:10px; align-items:center; margin:6px 0; font-size:13px; }
.bar-track { background:var(--line); border-radius:4px; height:8px; overflow:hidden; }
.bar-fill { display:block; height:100%; border-radius:4px; }
.bar-value { text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); }
ul.insights { margin:0; padding-left:18px; }
ul.insights li { margin:4px 0; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; }
td.num { font-variant-numeric:tabular-nums; }
.sev-critical { color:#d64545; font-weight:600; }
.sev-warning { color:#d97706; font-weight:600; }
.muted { color:var(--muted); }
footer { margin-top:28px; color:var(--muted); font-size:12px; text-align:center; }
"""


def _legend(items: List[Tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{c}"></i>{html.escape(n)}</span>' for n, c in items) + "</div>"


def render_report(data: Dict[str, Any]) -> str:
    m = data["metrics"]
    events = data["events"]
    summary = data["summary"]
    stats = compute_stats(data)
    t0 = m[0]["t"] if m else 0.0

    ts = [r["t"] for r in m]
    x_span = (ts[0], ts[-1]) if len(ts) > 1 else None
    present = [r for r in m if r.get("face")]
    bands = [(a, b, "band_calibration") for a, b in calibration_spans(m)]
    bands += [(e["start"], e.get("end") or e["start"], BAND_EVENTS[e["kind"]])
              for e in events if e.get("kind") in BAND_EVENTS]

    attention_chart = svg_line_chart(
        [("Attention", ts, [r["attention"] for r in m], PALETTE["attention"])],
        (0, 100), bands=bands, y_label="score")
    # Rate estimates need some history before they are meaningful.
    eye_rows = [r for r in present if r.get("perclos_ready", True) and r["t"] - t0 >= 20.0]
    eye_peak = max([r["blink_rate"] for r in eye_rows] + [100 * r["perclos"] for r in eye_rows], default=0.0)
    eyes_chart = svg_line_chart(
        [("Blink rate (/min)", [r["t"] for r in eye_rows], [r["blink_rate"] for r in eye_rows], PALETTE["blink"]),
         ("PERCLOS (%)", [r["t"] for r in eye_rows], [100 * r["perclos"] for r in eye_rows], PALETTE["perclos"])],
        (0, max(40.0, eye_peak * 1.1)), x_range=x_span)
    pose_chart = svg_line_chart(
        [("Yaw", [r["t"] for r in present], [r["yaw"] for r in present], PALETTE["yaw"]),
         ("Pitch", [r["t"] for r in present], [r["pitch"] for r in present], PALETTE["pitch"])],
        (-45, 45), y_label="degrees (camera frame)", x_range=x_span)

    posture_total = sum(stats["posture"].values()) or 1
    posture_items = [(k, 100.0 * v / posture_total) for k, v in stats["posture"].most_common()]
    gesture_items = [(k.replace("_", " "), float(v)) for k, v in stats["gestures"].most_common()]

    kpis = [
        (_fmt_time(stats["duration"]), "Duration"),
        (f"{stats['attention_mean']:.0f}", "Avg attention"),
        (f"{stats['focus_ratio'] * 100:.0f}%", "On-screen time"),
        (f"{stats['blinks']}", "Blinks"),
        (f"{stats['blink_rate_mean']:.1f}", "Blinks / min"),
        (f"{stats['perclos_peak'] * 100:.0f}%", "Peak PERCLOS"),
        (f"{stats['distractions']}", "Distractions"),
        (f"{stats['yawns']}", "Yawns"),
    ]
    kpi_html = "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{lbl}</div></div>' for v, lbl in kpis)

    rows = []
    for e in events:
        sev = e.get("severity", "info")
        rows.append(
            f"<tr><td class='num'>{_fmt_time(e['start'] - t0)}</td>"
            f"<td class='sev-{html.escape(sev)}'>{html.escape(sev.title())}</td>"
            f"<td>{html.escape(e.get('message') or e.get('kind', ''))}</td>"
            f"<td class='num'>{e.get('duration', 0.0):.1f} s</td></tr>")
    events_html = ("<table><thead><tr><th>Time</th><th>Severity</th><th>Event</th><th>Duration</th></tr></thead>"
                   f"<tbody>{''.join(rows)}</tbody></table>") if rows else '<p class="muted">No events.</p>'

    started = html.escape(_fmt_datetime(summary.get("started_at")))
    insights_html = "".join(f"<li>{html.escape(n)}</li>" for n in stats["insights"])

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Session Report - {html.escape(data['name'])}</title>
<style>{CSS}</style></head>
<body><main>
<header><h1>Behavior Session Report</h1>
<p>Session {html.escape(data['name'])}{' &middot; started ' + started if started else ''} &middot; {stats['samples']} samples</p></header>
<div class="kpis">{kpi_html}</div>
<section><h2>Key observations</h2><ul class="insights">{insights_html}</ul></section>
<section><h2>Attention over time</h2>{attention_chart}
{_legend([("Attention", PALETTE["attention"]), ("Calibration", "rgba(120,130,150,0.5)"), ("Distracted / absent", "rgba(217,95,2,0.5)"), ("Drowsy / microsleep", "rgba(200,30,45,0.6)")])}</section>
<section><h2>Eye metrics</h2>{eyes_chart}
{_legend([("Blink rate (/min)", PALETTE["blink"]), ("PERCLOS (%)", PALETTE["perclos"])])}</section>
<section><h2>Head orientation</h2>{pose_chart}
{_legend([("Yaw", PALETTE["yaw"]), ("Pitch", PALETTE["pitch"])])}</section>
<div class="two">
<section><h2>Posture (share of time)</h2>{svg_bar_list(posture_items, "#2f7de1", "{:.0f}%")}</section>
<section><h2>Gestures (occurrences)</h2>{svg_bar_list(gesture_items, "#1a9e77")}</section>
</div>
<section><h2>Event log</h2>{events_html}</section>
<footer>Generated by HBTS - Human Behavior Tracking System</footer>
</main></body></html>
"""


def generate_report(session_dir: str, output: Optional[str] = None) -> Path:
    session_path = Path(session_dir)
    if not session_path.is_dir():
        raise FileNotFoundError(f"session folder not found: {session_path}")
    data = load_session(session_path)
    out = Path(output) if output else session_path / "report.html"
    out.write_text(render_report(data), encoding="utf-8")
    return out
