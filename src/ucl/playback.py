"""Recorded episode inspection. This module never imports an environment or a model."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_FRAMES_BYTES = 96 * 2**20
# MLflow artifact mirrors, job logs and source snapshots are copies, never separate experiments.
EXCLUDED_RUN_PARTS = frozenset({"mlartifacts", "mlruns-isolated", "ui_jobs", "source_snapshot"})


@dataclass(frozen=True)
class Recording:
    path: Path
    metadata: dict
    frames: np.ndarray

    @property
    def length(self) -> int:
        return len(self.frames)


def confined_path(path: str | Path, root: str | Path) -> Path:
    """Reject links/path traversal outside the selected laboratory directory."""
    value, boundary = Path(path).resolve(), Path(root).resolve()
    if not value.is_relative_to(boundary):
        raise ValueError(f"Path is outside the laboratory: {path}")
    return value


def read_json(path: str | Path, max_bytes: int = 2 * 2**20) -> dict:
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError(f"JSON record is too large: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def load_recording(path: str | Path, root: str | Path) -> Recording:
    path = confined_path(path, root)
    metadata = read_json(path)
    for field in ("task_id", "seed", "architecture_id", "steps", "reward", "success"):
        if field not in metadata:
            raise ValueError(f"Incomplete episode: missing {field}")
    steps = metadata["steps"]
    if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
        raise ValueError("Episode steps must be a list of objects")
    frames_path = confined_path(
        path.parent / metadata.get("frames_file", "frames.npz"), path.parent
    )
    checksum = metadata.get("frames_sha256")
    if not checksum:
        raise ValueError("Missing frames_sha256: recording integrity is unverified")
    if frames_path.stat().st_size > MAX_FRAMES_BYTES:
        raise ValueError("Recording exceeds the replay memory limit")
    with frames_path.open("rb") as handle:
        actual_checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual_checksum != checksum:
        raise ValueError("Frame checksum mismatch; recording was changed")
    with zipfile.ZipFile(frames_path) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_FRAMES_BYTES:
            raise ValueError("Uncompressed frames exceed the replay memory limit")
    with np.load(frames_path, allow_pickle=False) as data:
        if set(data.files) != {"frames"}:
            raise ValueError("Expected a frames-only NPZ archive")
        frames = data["frames"]
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Recorded frames must be uint8 RGB images")
    if len(frames) != len(steps) + 1 or min(frames.shape[1:3]) < 1:
        raise ValueError("A replay needs one initial frame plus one frame per action")
    frames.setflags(write=False)
    return Recording(path, metadata, frames)


def synchronized_indices(lengths: list[int], step: int) -> list[int]:
    """Use one environment-step index. Shorter episodes remain on their final frame."""
    if not lengths or any(type(n) is not int or n < 1 for n in lengths):
        raise ValueError("Recordings must each contain an initial frame")
    if type(step) is not int or step < 0:
        raise ValueError("Playback step must be a nonnegative integer")
    return [min(step, length - 1) for length in lengths]


def step_details(recording: Recording, step: int) -> dict:
    index = synchronized_indices([recording.length], step)[0]
    details = {} if index == 0 else dict(recording.metadata["steps"][index - 1])
    return {
        "frame_index": index,
        "finished": index == recording.length - 1,
        "signals": details.get("signals", {}),
        "transition": details,
    }


def next_playback_step(
    step: int,
    last_step: int,
    playing: bool,
    elapsed: float,
    speed: float,
    base_steps_per_second: float = 2.0,
) -> tuple[int, bool]:
    if elapsed < 0 or speed <= 0 or last_step < 0 or step < 0:
        raise ValueError("Invalid playback state")
    if not playing:
        return min(step, last_step), False
    advance = max(0, math.floor(elapsed * speed * base_steps_per_second))
    current = min(step + advance, last_step)
    return current, current < last_step


def discover_architecture_configs(root: str | Path) -> tuple[list[Path], list[dict]]:
    """Find architecture configurations by their contract, never by a fixed ID list."""
    root = Path(root).resolve()
    available, unavailable = [], []
    for path in sorted((root / "configs").glob("*.json")):
        try:
            path = confined_path(path, root / "configs")
            config = read_json(path)
        except (OSError, ValueError) as exc:
            unavailable.append({"name": path.stem, "reason": str(exc)})
            continue
        if not isinstance(config.get("architecture"), dict) or not config.get("architecture_id"):
            continue
        if config.get("availability") == "unavailable":
            unavailable.append(
                {
                    "name": config["architecture_id"],
                    "reason": config.get("unavailable_reason", "No disponible"),
                }
            )
        else:
            available.append(path)
    return available, unavailable


def run_files(root: str | Path, pattern: str) -> list[Path]:
    """Original run files only; MLflow artifact mirrors are not separate experiments."""
    root = Path(root).resolve()
    files = []
    for path in (root / "runs").glob(pattern):
        path = confined_path(path, root / "runs")
        if not EXCLUDED_RUN_PARTS.intersection(path.relative_to(root / "runs").parts):
            files.append(path)
    return sorted(files)


def discover_recordings(root: str | Path) -> tuple[list[dict], list[str]]:
    root = Path(root).resolve()
    found, errors = [], []
    for path in run_files(root, "**/episode.json"):
        try:
            path = confined_path(path, root)
            value = read_json(path)
            environment = value.get("environment", {})
            task_sha256 = environment.get("task_sha256") if isinstance(environment, dict) else None
            found.append(
                {
                    "path": str(path),
                    "architecture_id": value["architecture_id"],
                    "task_id": value["task_id"],
                    "task_sha256": task_sha256,
                    "split": value.get("split"),
                    "seed": value["seed"],
                    "reward": value.get("reward"),
                    "success": value.get("success"),
                    "study": value.get("study") if isinstance(value.get("study"), dict) else None,
                }
            )
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{path.name}: {exc}")
    return found, errors


def comparable_recordings(a: dict, b: dict) -> bool:
    """Match a recorded task and seed; absent task provenance cannot establish equivalence."""
    task_hash = a.get("task_sha256")
    return (
        isinstance(task_hash, str)
        and bool(task_hash.strip())
        and isinstance(a.get("task_id"), str)
        and bool(a["task_id"])
        and type(a.get("seed")) is int
        and type(b.get("seed")) is int
        and all(a.get(key) == b.get(key) for key in ("task_id", "seed", "task_sha256"))
    )


def read_metrics(path: str | Path, limit: int = 2000) -> list[dict]:
    from collections import deque

    path = Path(path)
    if not path.exists():
        return []
    with path.open() as handle:
        lines = deque(handle, maxlen=limit)
    rows = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue  # A running writer may not have finished its last JSONL line.
        if isinstance(value, dict):
            rows.append(value)
    return rows
