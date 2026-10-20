"""Re-evaluate a finished study run with fresh sampling seeds (C08: how much is evaluation noise).

The study measures each checkpoint once, with the map seed as sampling seed. Here every
evaluation checkpoint of the learning curve (validation panel) and every matrix row (test panel)
is played again ``replicas`` times, with weights fixed and one new sampling seed per map and
replica: ``map_seed + replica * SAMPLING_OFFSET`` (replica 1, 2, ...; replica 0 would be the
study's own measurement). Per-episode outcomes are kept so variance can be split by seed, map
and sampling replica.

It is started for one run (its own event), never by scanning ``runs/``. Results go to
``runs/reevaluations/<run id>/reevaluation.json``; the study run itself is never modified.
A failure is written as ``status: FAILED`` with its error and can be retried by running it again.
"""

from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path

from .study_format import read_definition, resolve_study_run, run_id_of
from .study_recording import _sha256, run_definition

SAMPLING_OFFSET = 10_000_019
OUTPUT_DIR = "reevaluations"
MAX_REPLICAS = 20


def sampling_seed(map_seed: int, replica: int) -> int:
    if replica < 1:
        raise ValueError("Replicas start at 1; replica 0 is the study's own measurement")
    return int(map_seed) + int(replica) * SAMPLING_OFFSET


def output_path(root, run_id: str) -> Path:
    return Path(root).resolve() / "runs" / OUTPUT_DIR / run_id / "reevaluation.json"


def _checked_checkpoint(directory: Path, entry: dict) -> Path:
    path = (directory / entry["path"]).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError(f"Checkpoint file is missing or outside its run: {entry['path']}")
    if _sha256(path) != entry["sha256"]:
        raise ValueError(f"Checkpoint changed after the run saved it: {entry['path']}")
    return path


def _play(model, task, seeds, replicas, architecture_id):
    from .evaluation import evaluate_episode

    rows = []
    for replica in range(1, replicas + 1):
        outcomes = [evaluate_episode(model, task, seed, architecture_id=architecture_id,
                                     sampling_seed=sampling_seed(seed, replica))
                    for seed in seeds]
        rows.append({"replica": replica,
                     "success": [int(bool(r["success"])) for r in outcomes],
                     "success_rate": sum(bool(r["success"]) for r in outcomes) / len(outcomes),
                     "environment_steps": sum(len(r["steps"]) for r in outcomes)})
    return rows


def reevaluate_run(root, run_id: str, replicas: int = 3) -> dict:
    """Re-evaluate one finished study run; returns what was written to reevaluation.json."""
    from .checkpoints import load_checkpoint

    replicas = int(replicas)
    if not 1 <= replicas <= MAX_REPLICAS:
        raise ValueError(f"Replicas must be between 1 and {MAX_REPLICAS}")
    root = Path(root).resolve()
    directory, summary = resolve_study_run(root, run_id)
    if summary.get("status") != "FINISHED":
        raise ValueError("Only finished study runs can be re-evaluated")
    curve_points = [(key, point) for key, points in summary.get("curves", {}).items()
                    for point in points]
    if not curve_points or any("checkpoint" not in point for _, point in curve_points):
        raise ValueError("This run did not keep a checkpoint per evaluation; "
                         "re-evaluation needs one")
    run_id = run_id_of(root, directory)
    target = output_path(root, run_id)
    if target.is_file() and json.loads(target.read_text()).get("status") != "FAILED":
        raise FileExistsError(f"Already re-evaluated: {target}")
    definition = run_definition(root, summary)
    if {"validation", "test"} & set(definition["sealed_panels"]):
        raise ValueError("Validation and test panels must be open to re-evaluate")
    tasks = read_definition(definition["directory"], validate_tasks=True)["tasks"]
    task_of = {key: tasks[task_id] for key, task_id in definition["task_ids"].items()}
    seeds_of = {key: definition["panels"][task_id]
                for key, task_id in definition["task_ids"].items()}
    evaluation_entries = {entry["path"]: entry
                          for entry in summary.get("evaluation_checkpoints", {}).values()}
    architecture_id = summary["architecture_id"]
    result = {"kind": "study_reevaluation", "status": "RUNNING", "study_id": summary["study_id"],
              "protocol_hash": summary["protocol_hash"], "source_run": run_id,
              "architecture_id": architecture_id, "init_seed": summary.get("init_seed"),
              "task_keys": summary.get("task_keys"), "replicas": replicas,
              "sampling_rule": f"map_seed + replica * {SAMPLING_OFFSET}, replica = 1..{replicas}",
              "curves": {}, "matrix": {}, "parameters_updated": False}
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    lock = (root / "runs/.execution.lock").open("a")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another laboratory execution is active; re-evaluate after it") from exc
        try:
            for key, point in curve_points:
                entry = evaluation_entries.get(point["checkpoint"])
                if entry is None:
                    raise ValueError(
                        f"Curve checkpoint not listed in the summary: {point['checkpoint']}")
                path = _checked_checkpoint(directory, entry)
                model, _, _ = load_checkpoint(path, expected_architecture=summary["architecture"])
                seeds = seeds_of[key]["validation"]
                result["curves"].setdefault(key, []).append({
                    "step": point["step"], "checkpoint": entry["path"],
                    "sha256": entry["sha256"], "panel": "validation", "seeds": seeds,
                    "study_success_rate": point["success_rate"],
                    "replicas": _play(model, task_of[key], seeds, replicas, architecture_id)})
            for name, row in summary.get("matrix", {}).items():
                entry = summary["checkpoints"][name]
                path = _checked_checkpoint(directory, entry)
                model, _, _ = load_checkpoint(path, expected_architecture=summary["architecture"])
                result["matrix"][name] = {
                    key: {"checkpoint": entry["path"], "sha256": entry["sha256"],
                          "panel": "test", "seeds": seeds_of[key]["test"],
                          "study_success_rate": row["test"][key]["success_rate"],
                          "replicas": _play(model, task_of[key], seeds_of[key]["test"],
                                            replicas, architecture_id)}
                    for key in sorted(row["test"])}
            result.update(status="FINISHED", elapsed_seconds=time.perf_counter() - started)
        except Exception as exc:
            result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                          elapsed_seconds=time.perf_counter() - started)
            target.write_text(json.dumps(result, indent=1, allow_nan=False))
            raise
        target.write_text(json.dumps(result, indent=1, allow_nan=False))
        return result
    finally:
        lock.close()
