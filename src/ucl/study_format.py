"""What makes a folder a study and a run a study run. JSON only: no simulator or model imports.

A **study definition** is any folder ``configs/<name>/`` holding three frozen files:

- ``protocol.json``: ``study_id`` and ``tasks`` (task key -> source task). Optional
  ``sealed_panels`` (default ``["final"]``): panels that stay sealed until the study's own final
  evaluation; the visual lab never records them.
- ``tasks.json``: the task manifest, with the same contract as ``configs/tasks.json``.
- ``panels.json``: ``task_ids`` (task key -> task id) and ``panels`` (task id -> panel name ->
  list of map seeds).

Its ``protocol_hash`` is the sha256 of the three files' bytes in that order, the same value a
study run records. A **study run** is any folder under ``runs/`` whose ``summary.json`` carries
``study_id`` and ``protocol_hash``; it belongs to the definition with the same hash.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

STUDY_FILES = ("protocol.json", "tasks.json", "panels.json")
DEFAULT_SEALED = ("final",)
PANEL_PREFERENCE = ("validation", "test", "known", "train")
MAX_JSON_BYTES = 16 * 2**20
# Task keys and panel names become folder names of recordings, so they stay plain words.
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _read(path: Path) -> dict:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"{path.name} is too large")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def protocol_hash(directory) -> str:
    directory = Path(directory)
    return hashlib.sha256(
        b"".join((directory / name).read_bytes() for name in STUDY_FILES)
    ).hexdigest()


def read_definition(directory, *, validate_tasks: bool = False) -> dict:
    """Read one study definition. ``validate_tasks`` also checks the task manifest contract."""
    directory = Path(directory)
    protocol, manifest, panels_file = (_read(directory / name) for name in STUDY_FILES)
    study_id = protocol.get("study_id")
    if not isinstance(study_id, str) or not study_id.strip():
        raise ValueError("protocol.json needs a study_id")
    keys = protocol.get("tasks")
    if not isinstance(keys, dict) or not keys:
        raise ValueError("protocol.json needs a non-empty tasks object")
    if validate_tasks:
        from .tasks import validate_manifest

        validate_manifest(manifest)
    tasks, task_ids = manifest.get("tasks"), panels_file.get("task_ids")
    panels = panels_file.get("panels")
    if (
        not isinstance(tasks, dict)
        or not isinstance(task_ids, dict)
        or not isinstance(panels, dict)
    ):
        raise ValueError("tasks.json needs tasks; panels.json needs task_ids and panels")
    if set(task_ids) != set(keys):
        raise ValueError("panels.json task_ids must use the protocol task keys")
    for key, task_id in task_ids.items():
        if not SAFE_NAME.match(str(key)):
            raise ValueError(f"Task key {key!r} must be a plain name (letters, digits, _ . -)")
        if not isinstance(task_id, str) or task_id not in tasks:
            raise ValueError(f"Task {key} ({task_id}) is missing from tasks.json")
        if not isinstance(tasks[task_id], dict) or not isinstance(panels.get(task_id), dict):
            raise ValueError(f"Task {key} ({task_id}) is missing from tasks.json or panels.json")
        for name, seeds in panels[task_id].items():
            if not SAFE_NAME.match(name):
                raise ValueError(f"Panel {name!r} must be a plain name (letters, digits, _ . -)")
            if not isinstance(seeds, list) or any(type(seed) is not int for seed in seeds):
                raise ValueError(f"Panel {name} of task {key} must be a list of integer seeds")
    sealed = protocol.get("sealed_panels", list(DEFAULT_SEALED))
    if not isinstance(sealed, list) or any(not isinstance(name, str) for name in sealed):
        raise ValueError("sealed_panels must be a list of panel names")
    return {
        "study_id": study_id,
        "name": directory.name,
        "directory": directory,
        "protocol": protocol,
        "tasks": tasks,
        "task_ids": task_ids,
        "panels": panels,
        "sealed_panels": sealed,
        "protocol_hash": protocol_hash(directory),
    }


def discover_studies(root) -> tuple[list[dict], list[str]]:
    """Every ``configs/<name>/`` that holds the three study files, by format, never by name."""
    base = (Path(root).resolve() / "configs").resolve()
    found, errors = [], []
    if not base.is_dir():
        return found, errors
    for directory in sorted(path.parent for path in base.glob("*/protocol.json")):
        if not all((directory / name).is_file() for name in STUDY_FILES):
            continue
        try:
            if directory.resolve().parent != base:
                raise ValueError("the folder points outside configs/")
            found.append(read_definition(directory))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append(f"No se pudo leer el estudio de configs/{directory.name}: {exc}")
    return found, errors


def definition_for(summary: dict, definitions: list[dict]) -> dict | None:
    """The definition a run was trained against: same study and the same frozen files."""
    return next(
        (
            item
            for item in definitions
            if item["study_id"] == summary.get("study_id")
            and item["protocol_hash"] == summary.get("protocol_hash")
        ),
        None,
    )


def panel_order(definition: dict, task_key: str) -> list[str]:
    """Panels that can be recorded for a task: never the sealed ones, usual panels first."""
    names = [
        name
        for name in definition["panels"][definition["task_ids"][task_key]]
        if name not in definition["sealed_panels"]
    ]
    known = [name for name in PANEL_PREFERENCE if name in names]
    return known + [name for name in names if name not in known]


def study_run_files(root) -> list[Path]:
    """``summary.json`` files under ``runs/`` (MLflow mirrors excluded) that declare a study."""
    from .playback import run_files

    found = []
    for path in run_files(root, "**/summary.json"):
        try:
            if path.stat().st_size <= MAX_JSON_BYTES and '"study_id"' in path.read_text():
                found.append(path)
        except OSError:
            continue
    return found


def run_id_of(root, directory) -> str:
    """A study run is named by its folder relative to ``runs/``, e.g. ``study_001/<folder>``."""
    return Path(directory).resolve().relative_to(Path(root).resolve() / "runs").as_posix()


def resolve_study_run(root, run_id: str) -> tuple[Path, dict]:
    """(directory, summary) of a finished study run.

    ``run_id`` is the folder relative to ``runs/``. A bare folder name (older links) is accepted
    only when exactly one study run has it.
    """
    root = Path(root).resolve()
    base = (root / "runs").resolve()
    text = str(run_id or "").strip()
    directory = (base / text).resolve()
    if not text or directory == base or not directory.is_relative_to(base):
        raise ValueError(f"Not a study run: {run_id}")
    if not (directory / "summary.json").is_file() and "/" not in text and "\\" not in text:
        matches = [path.parent for path in study_run_files(root) if path.parent.name == text]
        if len(matches) > 1:
            raise ValueError(f"Several study runs are called {text}; use its path under runs/")
        if len(matches) == 1:
            directory = matches[0].resolve()
    from .playback import EXCLUDED_RUN_PARTS

    if EXCLUDED_RUN_PARTS.intersection(directory.relative_to(base).parts):
        raise ValueError(f"Not a study run: {run_id}")
    if not (directory / "summary.json").is_file():
        raise ValueError(f"Not a study run: {run_id}")
    summary = _read(directory / "summary.json")
    if not summary.get("study_id") or not summary.get("protocol_hash"):
        raise ValueError(f"Not a study run summary: {run_id}")
    if summary.get("status") != "FINISHED":
        raise ValueError(f"Run did not finish ({summary.get('status')}): {run_id}")
    return directory, summary
