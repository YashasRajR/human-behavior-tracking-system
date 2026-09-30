# Architecture

HBTS is organised as a one-directional pipeline. Each stage only depends on
the stages before it, which keeps the analytics testable without a camera and
lets the UI be replaced without touching the maths.

```
 VideoSource ──> PerceptionPipeline ──> BehaviorEngine ──> Dashboard
  (io/source)      (vision/*)            (analytics/*)      (ui/*)
                        │                     │
                  FrameAnalysis          BehaviorState ──> SessionRecorder ──> HTML report
                  (core/types)                               (io/recorder)      (io/report)
```

## Layers

| Package | Responsibility | Depends on |
| --- | --- | --- |
| `hbts.core` | Geometry (EAR, MAR, iris ratio, PnP head pose), signal filters, shared data types, landmark index tables | NumPy (OpenCV only for `solvePnP`) |
| `hbts.vision` | Runs MediaPipe Tasks models and converts their output into `FaceObservation`, `HandObservation`, `PoseObservation` | MediaPipe, core |
| `hbts.analytics` | Stateful, time-aware interpretation: blinks, PERCLOS, yawns, calibration, attention, events, gesture commands | core |
| `hbts.io` | Camera / video input, model download, session recording, report generation | core, analytics |
| `hbts.ui` | OpenCV dashboard: theme, primitives, overlays, layout | core, analytics |
| `hbts.app`, `hbts.cli` | Wiring, keyboard handling, command-line interface | everything |

## Design decisions

**Explicit timestamps everywhere.** Every analytics component receives `t`
(seconds) as an argument instead of calling `time.time()`. Live cameras use
wall-clock time; video files use `frame_index / fps`. The same code therefore
produces identical results on recorded video, in real time, and in unit tests
driven by synthetic signals.

**Pure feature extraction.** `extract_face_features` and
`extract_pose_features` are pure functions of a landmark array. Tests run them
on recorded landmarks (`tests/data/`) without loading any model.

**Relative, calibrated thresholds.** Absolute thresholds (for example,
"EAR < 0.15 means closed") do not transfer between people or camera
placements. A short calibration phase records personal baselines, and every
decision is made relative to them. See [methodology.md](methodology.md).

**Debounced events.** Per-frame conditions are converted into events by rules
with onset and offset durations, so a single dropped detection does not
register as "user left".

**Non-blocking I/O.** The recorder writes on a background thread through a
bounded queue; the video loop never waits for the disk.

**Privacy by design.** Privacy mode replaces the camera image with a neutral
canvas before anything is drawn, so neither the on-screen view, snapshots nor
exported video contain camera pixels. Sessions store only derived metrics,
never frames (unless a snapshot is explicitly requested).

## Data model

```
FrameAnalysis            one frame of perception
  face:  FaceObservation   EAR (L/R), MAR, gaze ratio (h, v), pitch/yaw/roll, landmarks
  hands: [HandObservation] handedness, gesture, score, landmarks, bbox
  pose:  PoseObservation   shoulder tilt, neck angle, head drop ratio

BehaviorState            one frame of interpretation (UI + recorder)
  attention, on_screen, focus_ratio, blink_count, blink_rate, perclos,
  fatigue, head/gaze labels, posture, active events, command progress ...
```

## Session folder

```
sessions/<YYYYmmdd-HHMMSS>/
  config.json      resolved configuration
  metrics.jsonl    10 Hz time series (BehaviorState.metrics())
  metrics.csv      same data as CSV
  events.jsonl     completed events: kind, start, end, duration, severity, message
  summary.json     totals and calibration baseline
  report.html      self-contained report (inline SVG, no external assets)
  snapshots/       frames saved on request
```
