"""Record a study's episodes from a saved checkpoint so the visual lab can replay them.

Works for any study that follows ``ucl.study_format``. Weights stay fixed and sealed panels stay
sealed. Each map is played with the sampling rule of the study's panel evaluation (a generator
seeded with the map seed), so a recording reproduces the episode the study counted for that
checkpoint and map. Output lives under ``runs/`` next to other dashboard evaluations; study runs
are never modified.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

from .study_format import (
    definition_for,
    discover_studies,
    panel_order,
    read_definition,
    resolve_study_run,
    run_id_of,
)

MAX_MAPS = 16


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run_definition(root, summary: dict) -> dict:
    """The study definition in ``configs/`` that this run was trained against."""
    definitions, _ = discover_studies(root)
    definition = definition_for(summary, definitions)
    if definition is not None:
        return definition
    if any(item["study_id"] == summary.get("study_id") for item in definitions):
        raise ValueError("The study protocol changed after this run; its panels no longer match")
    raise ValueError(f"No study definition in configs/ matches {summary.get('study_id')}")


def checked_output(root: Path, output) -> Path:
    """A new folder under ``runs/`` that is not inside any run, its evidence or an MLflow copy."""
    from .playback import EXCLUDED_RUN_PARTS

    base = (root / "runs").resolve()
    output = Path(output).resolve()
    if not output.is_relative_to(base) or output == base:
        raise ValueError("Recording output must be a new folder inside runs/")
    if EXCLUDED_RUN_PARTS.intersection(output.relative_to(base).parts):
        raise ValueError("Recording output cannot go inside an MLflow copy or a job log folder")
    for folder in (output, *output.parents):
        if folder == base:
            break
        if (folder / "summary.json").exists() or (folder / "recording.json").exists():
            raise ValueError("Recording output cannot go inside another run or recording")
    return output


def check_request(definition: dict, task_key: str, panel: str, maps) -> tuple[str, list, list]:
    """Return (task_id, panel seeds, 1-based map numbers) or raise before anything runs."""
    if panel in definition["sealed_panels"]:
        raise ValueError("This panel stays sealed until the study's final evaluation")
    if task_key not in definition["task_ids"]:
        raise ValueError(f"Unknown study task: {task_key}")
    if panel not in panel_order(definition, task_key):
        raise ValueError(f"Unknown panel: {panel}")
    try:
        numbers = [int(value) for value in maps]
    except (TypeError, ValueError) as exc:
        raise ValueError("Maps must be whole numbers") from exc
    if not numbers or len(numbers) > MAX_MAPS or len(set(numbers)) != len(numbers):
        raise ValueError(f"Choose between 1 and {MAX_MAPS} distinct maps")
    task_id = definition["task_ids"][task_key]
    seeds = definition["panels"][task_id][panel]
    if any(not 1 <= number <= len(seeds) for number in numbers):
        raise ValueError(f"Maps must be between 1 and {len(seeds)} for the {panel} panel")
    return task_id, seeds, numbers


def record_study_episodes(
    root, run_id: str, checkpoint: str, task_key: str, panel: str, maps, output
) -> dict:
    """Replay ``checkpoint`` of a study run on numbered panel maps (1-based) and save frames."""
    from .checkpoints import load_checkpoint
    from .evaluation import evaluate_episode

    root = Path(root).resolve()
    directory, summary = resolve_study_run(root, run_id)
    definition = run_definition(root, summary)
    task_id, seeds, numbers = check_request(definition, task_key, panel, maps)
    entry = summary.get("checkpoints", {}).get(checkpoint)
    if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
        raise ValueError(f"Checkpoint {checkpoint!r} is not recorded by this run")
    path = (directory / entry["path"]).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError("Checkpoint file is missing or outside its run")
    if _sha256(path) != entry["sha256"]:
        raise ValueError("Checkpoint changed after the run saved it")
    output = checked_output(root, output)
    # The task manifest contract is checked here, where an environment will be built from it.
    task = read_definition(definition["directory"], validate_tasks=True)["tasks"][task_id]
    source_run = run_id_of(root, directory)

    lock = (root / "runs/.execution.lock").open("a")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another laboratory execution is active; record when it finishes"
            ) from exc
        model, _, _ = load_checkpoint(path, expected_architecture=summary["architecture"])
        output.mkdir(parents=True, exist_ok=False)
        episodes = []
        for number in numbers:
            seed = int(seeds[number - 1])
            episode_dir = output / "episodes" / f"{task_key}-{panel}-{number:02d}"
            record = evaluate_episode(
                model, task, seed, output=episode_dir, architecture_id=summary["architecture_id"]
            )
            context = {
                "study_id": summary["study_id"],
                "study_definition": definition["name"],
                "protocol_hash": summary["protocol_hash"],
                "source_run": source_run,
                "phase": summary.get("phase"),
                "init_seed": summary.get("init_seed"),
                "trained_tasks": summary.get("task_keys"),
                "checkpoint": checkpoint,
                "checkpoint_sha256": entry["sha256"],
                "task_key": task_key,
                "panel": panel,
                "map_number": number,
                "sampling_rule": "map seed, as in the study's panel evaluation",
                "mlflow": summary.get("mlflow", {}),
            }
            episode_path = episode_dir / "episode.json"
            saved = json.loads(episode_path.read_text())
            saved["study"] = context
            episode_path.write_text(json.dumps(saved, indent=2, allow_nan=False))
            episodes.append(
                {
                    "map_number": number,
                    "seed": seed,
                    "success": record["success"],
                    "reward": record["reward"],
                    "actions": len(record["steps"]),
                    "episode": str(episode_path.relative_to(root)),
                }
            )
        result = {
            "kind": "study_recording",
            "status": "FINISHED",
            "study_id": summary["study_id"],
            "architecture_id": summary["architecture_id"],
            "source_run": source_run,
            "checkpoint": checkpoint,
            "task_key": task_key,
            "task_id": task_id,
            "panel": panel,
            "episodes": episodes,
            "parameters_updated": False,
            "mlflow": summary.get("mlflow", {}),
        }
        (output / "recording.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return result
    finally:
        lock.close()
