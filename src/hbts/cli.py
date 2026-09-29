"""Command-line interface.

    hbts                          run on the default webcam
    hbts run --video clip.mp4     analyse a recorded video
    hbts run --headless --video clip.mp4   batch mode, no window
    hbts report sessions/<id>     rebuild the HTML report for a session
    hbts models                   pre-download all model files
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from hbts import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hbts",
        description="Real-time human behavior tracking: attention, gaze, blinks, fatigue, gestures and posture.",
    )
    parser.add_argument("--version", action="version", version=f"hbts {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="start live tracking (default)")
    run.add_argument("--camera", type=int, default=None, help="webcam index (default from config: 0)")
    run.add_argument("--video", type=str, default=None, help="process a video file instead of a camera")
    run.add_argument("--config", type=str, default=None, help="YAML configuration file")
    run.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                     help="override a configuration value, e.g. --set attention.yaw_tolerance=25")
    run.add_argument("--no-face", action="store_true", help="disable face, eye and head analysis")
    run.add_argument("--no-hands", action="store_true", help="disable hand and gesture analysis")
    run.add_argument("--no-pose", action="store_true", help="disable posture analysis")
    run.add_argument("--no-record", action="store_true", help="do not write a session folder")
    run.add_argument("--privacy", action="store_true", help="start in privacy mode (camera image hidden)")
    run.add_argument("--headless", action="store_true", help="no window; useful with --video for batch runs")
    run.add_argument("--save-video", type=str, default=None, metavar="PATH",
                     help="also write the rendered dashboard to an .mp4 file")
    run.add_argument("--session-name", type=str, default=None, help="name for the session folder")

    report = sub.add_parser("report", help="generate the HTML report for a recorded session")
    report.add_argument("session", help="path to a session folder, e.g. sessions/20260930-142501")
    report.add_argument("-o", "--output", default=None, help="output file (default: <session>/report.html)")

    models = sub.add_parser("models", help="download all model files")
    models.add_argument("--dir", default="models", help="target directory (default: models)")
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    from hbts.config import load_config

    overrides = list(args.overrides)
    if args.camera is not None:
        overrides.append(f"source.camera={args.camera}")
    for flag, key in ((args.no_face, "face"), (args.no_hands, "hands"), (args.no_pose, "pose")):
        if flag:
            overrides.append(f"modules.{key}=false")
    if args.no_record:
        overrides.append("recording.enabled=false")
    if args.privacy:
        overrides.append("ui.privacy=true")
    if args.headless and not args.video:
        print("--headless requires --video", file=sys.stderr)
        return 2

    cfg = load_config(args.config, overrides)

    from hbts.app import Application
    from hbts.io.assets import ModelUnavailableError

    try:
        app = Application(cfg, video=args.video, headless=args.headless,
                          save_video=args.save_video, session_name=args.session_name)
    except (ModelUnavailableError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    app.run()
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from hbts.io.report import generate_report

    try:
        path = generate_report(args.session, args.output)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Report written to {path}")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    from hbts.config import DEFAULTS
    from hbts.io.assets import ModelUnavailableError, ensure_models

    names = [DEFAULTS["models"][k] for k in ("face", "hands", "pose")]
    try:
        paths = ensure_models(names, args.dir)
    except ModelUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for name, path in paths.items():
        print(f"  {name:<28} {path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help", "--version"):
        argv = ["run", *argv]
    args = _build_parser().parse_args(argv)
    handlers = {"run": _cmd_run, "report": _cmd_report, "models": _cmd_models}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
