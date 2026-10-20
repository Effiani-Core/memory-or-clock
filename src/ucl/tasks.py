"""Versioned, local task manifests. No benchmark downloads or model dependencies."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

DEFAULT_TASKS = Path(__file__).resolve().parents[2] / "configs" / "tasks.json"
SPLITS = ("development", "validation", "heldout")
TASK_FIELDS = (
    "adapter",
    "env_id",
    "height",
    "width",
    "view_size",
    "max_steps",
    "goal",
    "rules",
    "init_tiles",
)


def task_identity(task: dict) -> str:
    """Identity excludes labels/seeds, so aliases cannot conceal duplicate tasks."""
    payload = {key: task[key] for key in TASK_FIELDS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def ruleset_identity(task: dict) -> str:
    payload = {key: task[key] for key in ("goal", "rules", "init_tiles")}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_manifest(data: dict) -> None:
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported task manifest schema_version")
    tasks, splits = data.get("tasks", {}), data.get("splits", {})
    if not tasks or set(splits) != set(SPLITS):
        raise ValueError("Tasks and explicit development/validation/heldout splits are required")
    seen_ids, seen_hashes, ruleset_splits = set(), set(), {}
    seed_sets = {}
    for split, identifiers in splits.items():
        seed_sets[split] = set()
        for identifier in identifiers:
            if identifier in seen_ids or identifier not in tasks:
                raise ValueError(f"Duplicate or unknown task identity: {identifier}")
            task = tasks[identifier]
            if task.get("id") != identifier or task.get("split") != split:
                raise ValueError(f"Task/split mismatch: {identifier}")
            try:
                identity = task_identity(task)
                rule_identity = ruleset_identity(task)
            except KeyError as exc:
                raise ValueError(f"Missing task field: {exc}") from exc
            if identity != task.get("task_sha256") or rule_identity != task.get("ruleset_sha256"):
                raise ValueError(f"Task checksum mismatch: {identifier}")
            if identity in seen_hashes:
                raise ValueError(f"Duplicate task content: {identifier}")
            if rule_identity in ruleset_splits and ruleset_splits[rule_identity] != split:
                raise ValueError(f"Ruleset leaks across splits: {identifier}")
            ruleset_splits[rule_identity] = split
            seen_ids.add(identifier)
            seen_hashes.add(identity)
            seeds = task.get("seeds", [])
            if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
                raise ValueError(f"Invalid seed list: {identifier}")
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"Duplicate episode seed: {identifier}")
            evaluation_seeds = task.get("evaluation_seeds", [])
            if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in evaluation_seeds):
                raise ValueError(f"Invalid evaluation seed list: {identifier}")
            if len(set(evaluation_seeds)) != len(evaluation_seeds) or set(seeds) & set(
                evaluation_seeds
            ):
                raise ValueError(f"Evaluation/training seed overlap: {identifier}")
            seed_sets[split].update(seeds + evaluation_seeds)
    if seen_ids != set(tasks):
        raise ValueError("Every task must belong to exactly one split")
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1 :]:
            if seed_sets[left] & seed_sets[right]:
                raise ValueError("Seed lists must be disjoint between splits")
    for curriculum in data.get("curricula", {}).values():
        for stage in curriculum.get("stages", []):
            if stage.get("task_id") not in tasks:
                raise ValueError("Curriculum refers to an unknown task")
            task = tasks[stage["task_id"]]
            if task["split"] != "development" and stage.get("train_steps", 0) != 0:
                raise ValueError("Validation/heldout tasks cannot appear as training stages")
            if not isinstance(stage.get("train_steps", 0), int) or stage.get("train_steps", 0) < 0:
                raise ValueError("train_steps must be nonnegative")
            evaluation_seeds = stage.get("evaluation_seeds", [])
            if len(set(evaluation_seeds)) != len(evaluation_seeds):
                raise ValueError("Repeated evaluation seed in one stage")
            if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in evaluation_seeds):
                raise ValueError("Invalid evaluation seed")
            if stage.get("purpose") in {"retention", "new_maps"} and set(evaluation_seeds) & set(
                task["seeds"]
            ):
                raise ValueError("Retention/generalization evaluation seeds overlap training seeds")


def load_tasks(path: str | Path | None = None) -> dict:
    data = json.loads(Path(path or DEFAULT_TASKS).read_text())
    validate_manifest(data)
    return copy.deepcopy(data)
