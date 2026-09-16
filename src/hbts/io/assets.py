"""Download and cache the MediaPipe Tasks model bundles.

Model files are several megabytes each, so they are not committed to the
repository. The first run downloads them into ``models/`` (or the directory set
in the configuration); subsequent runs reuse the cached copies.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, Iterable

_BASE = "https://storage.googleapis.com/mediapipe-models"

MODEL_URLS: Dict[str, str] = {
    "face_landmarker.task": f"{_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "gesture_recognizer.task": f"{_BASE}/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
    "pose_landmarker_lite.task": f"{_BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "pose_landmarker_full.task": f"{_BASE}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
}


class ModelUnavailableError(RuntimeError):
    """Raised when a model is missing and cannot be downloaded."""


def _download(url: str, target: Path, quiet: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=target.parent, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp_path.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total and not quiet:
                    pct = 100.0 * done / total
                    sys.stdout.write(f"\r  {target.name:<28} {pct:5.1f}%  ({done / 1e6:.1f} MB)")
                    sys.stdout.flush()
        if not quiet:
            sys.stdout.write("\n")
        shutil.move(str(tmp_path), str(target))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def ensure_model(name: str, model_dir: str = "models", quiet: bool = False) -> Path:
    """Return the local path of ``name``, downloading it on first use."""
    target = Path(model_dir) / name
    if target.is_file() and target.stat().st_size > 0:
        return target

    url = MODEL_URLS.get(name)
    if url is None:
        raise ModelUnavailableError(
            f"'{name}' was not found in '{model_dir}' and has no known download URL."
        )
    if not quiet:
        print(f"Downloading model {name} ...")
    try:
        _download(url, target, quiet=quiet)
    except Exception as exc:  # network errors, proxies, permissions
        raise ModelUnavailableError(
            f"Could not download '{name}' ({exc}).\n"
            f"Download it manually from:\n  {url}\n"
            f"and place it at:\n  {target.resolve()}"
        ) from exc
    return target


def ensure_models(names: Iterable[str], model_dir: str = "models", quiet: bool = False) -> Dict[str, Path]:
    return {name: ensure_model(name, model_dir, quiet=quiet) for name in names}
