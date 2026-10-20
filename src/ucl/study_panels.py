"""STUDY-001 frozen task manifest and deduplicated map panels. Identities are evaluator-only."""

from . import runtime  # noqa: F401, I001 -- CPU JAX/offline defaults precede other imports.

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .envs import make_env
from .tasks import load_tasks, ruleset_identity, task_identity, validate_manifest

LAB_ROOT = Path(__file__).resolve().parents[2]
PANEL_ORDER = ("train", "validation", "test", "final")
PANEL_KEYS = PANEL_ORDER + ("known",)
GENERALIZATION_KIND = "study_001_same_rules_colors_new_layouts"
IDENTITY_DEFINITION = (
    "sha256 over the XLand initial state bytes of grid, agent position, agent direction and "
    "agent pocket (in that order); the seed is not part of the identity."
)
SCAN_LIMIT_FACTOR = 20


def physical_identity(env) -> str:
    """Hash of the initial physical layout. Privileged evaluator metadata, never a model input."""
    if getattr(env, "_timestep", None) is None:
        raise RuntimeError("Call reset before computing a physical identity")
    state = env._timestep.state
    digest = hashlib.sha256()
    for value in (state.grid, state.agent.position, state.agent.direction, state.agent.pocket):
        digest.update(np.ascontiguousarray(np.asarray(value)).tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_path(protocol: dict) -> Path:
    path = Path(protocol["source_manifest"])
    return path if path.is_absolute() else LAB_ROOT / path


def _study_task(source: dict, key: str, grid_size: int) -> dict:
    task = copy.deepcopy(source)
    task.update(
        id=f"study001-{key}-{task['family']}",
        height=grid_size,
        width=grid_size,
        split="development",
        generalization_kind=GENERALIZATION_KIND,
    )
    task["task_sha256"] = task_identity(task)
    task["ruleset_sha256"] = ruleset_identity(task)
    return task


def _scan_distinct(task: dict, base: int, needed: int) -> tuple[list[int], dict, int]:
    """First seed per new physical identity, walking base, base+1, ... (one-off generation)."""
    env = make_env(task)
    seeds, identities, seen = [], {}, set()
    scanned = 0
    try:
        while len(seeds) < needed:
            if scanned >= SCAN_LIMIT_FACTOR * needed:
                raise ValueError(
                    f"{task['id']}: only {len(seeds)} distinct maps in {scanned} seeds "
                    f"(needed {needed})"
                )
            seed = base + scanned
            scanned += 1
            env.reset(seed)
            identity = physical_identity(env)
            if identity in seen:
                continue
            seen.add(identity)
            seeds.append(seed)
            identities[str(seed)] = identity
    finally:
        env.close()
    return seeds, identities, scanned


def build_study(protocol_path, out_dir, *, force=False, panel_sizes=None) -> dict:
    protocol_path, out_dir = Path(protocol_path), Path(out_dir)
    protocol = json.loads(protocol_path.read_bytes())
    sizes = dict(protocol["panels"])
    sizes.update(panel_sizes or {})
    if sizes["known"] > sizes["train"]:
        raise ValueError("known panel cannot be larger than train")
    tasks_path, panels_path = out_dir / "tasks.json", out_dir / "panels.json"
    if not force and (tasks_path.exists() or panels_path.exists()):
        raise FileExistsError(f"Refusing to overwrite study files in {out_dir} (use force=True)")

    source = load_tasks(_source_path(protocol))
    needed = sum(sizes[name] for name in PANEL_ORDER)
    tasks, task_ids, panels, identities, scanned = {}, {}, {}, {}, {}
    for key in sorted(protocol["tasks"]):
        task = _study_task(source["tasks"][protocol["tasks"][key]], key, protocol["grid_size"])
        seeds, seed_identities, count = _scan_distinct(
            task, protocol["panel_seed_base"][key], needed
        )
        split, start = {}, 0
        for name in PANEL_ORDER:
            split[name] = seeds[start : start + sizes[name]]
            start += sizes[name]
        split["known"] = split["train"][: sizes["known"]]
        task["seeds"] = list(split["train"])
        task["evaluation_seeds"] = list(split["validation"])
        tasks[task["id"]] = task
        task_ids[key] = task["id"]
        panels[task["id"]] = split
        identities[task["id"]] = seed_identities
        scanned[task["id"]] = count

    manifest = {
        "schema_version": 1,
        "description": (
            f"{protocol['study_id']} frozen tasks on {protocol['grid_size']}x"
            f"{protocol['grid_size']} grids. Training seeds are the train panel and evaluation "
            "seeds the validation panel; test/final panels live in panels.json."
        ),
        "observation_policy": source["observation_policy"],
        "tasks": tasks,
        "splits": {"development": list(tasks), "validation": [], "heldout": []},
        "curricula": {},
    }
    validate_manifest(manifest)
    study = {
        "study_id": protocol["study_id"],
        "identity_definition": IDENTITY_DEFINITION,
        "protocol_sha256": _sha256_file(protocol_path),
        "task_ids": task_ids,
        "panels": panels,
        "identities": identities,
        "seeds_scanned": scanned,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text(json.dumps(manifest, indent=2) + "\n")
    panels_path.write_text(json.dumps(study, indent=2) + "\n")
    return study


def load_study(study_dir) -> dict:
    """Load and validate a frozen study. Raises ValueError on any inconsistency."""
    study_dir = Path(study_dir)
    paths = [study_dir / name for name in ("protocol.json", "tasks.json", "panels.json")]
    protocol = json.loads(paths[0].read_bytes())
    tasks = load_tasks(paths[1])["tasks"]
    study = json.loads(paths[2].read_bytes())

    if study.get("protocol_sha256") != _sha256_file(paths[0]):
        raise ValueError("protocol.json changed since the panels were built")
    task_ids = study.get("task_ids", {})
    if set(task_ids) != set(protocol["tasks"]) or set(task_ids.values()) != set(tasks):
        raise ValueError("task_ids do not match protocol tasks and tasks.json")
    if set(study.get("panels", {})) != set(tasks) or set(study.get("identities", {})) != set(tasks):
        raise ValueError("panels/identities must cover exactly the study tasks")
    sizes = protocol["panels"]
    for key, task_id in task_ids.items():
        if not task_id.startswith(f"study001-{key}-"):
            raise ValueError(f"task id {task_id} does not belong to task {key}")
        panel, identity = study["panels"][task_id], study["identities"][task_id]
        if set(panel) != set(PANEL_KEYS):
            raise ValueError(f"{task_id}: panels must be exactly {PANEL_KEYS}")
        for name in PANEL_KEYS:
            if len(panel[name]) != sizes[name] or len(set(panel[name])) != len(panel[name]):
                raise ValueError(f"{task_id}: {name} panel size does not match protocol")
        if panel["known"] != panel["train"][: sizes["known"]]:
            raise ValueError(f"{task_id}: known must be the first train seeds")
        seen_seeds, seen_identities = set(), set()
        for name in PANEL_ORDER:
            for seed in panel[name]:
                if seed in seen_seeds:
                    raise ValueError(f"{task_id}: seed {seed} appears in more than one panel")
                if str(seed) not in identity:
                    raise ValueError(f"{task_id}: seed {seed} has no recorded identity")
                if identity[str(seed)] in seen_identities:
                    raise ValueError(f"{task_id}: seed {seed} repeats a physical map")
                seen_seeds.add(seed)
                seen_identities.add(identity[str(seed)])
        if set(identity) != {str(seed) for seed in seen_seeds}:
            raise ValueError(f"{task_id}: identities do not match the panel seeds")
        task = tasks[task_id]
        if task["seeds"] != panel["train"] or task["evaluation_seeds"] != panel["validation"]:
            raise ValueError(f"{task_id}: task seeds must equal the train/validation panels")
        if task["height"] != protocol["grid_size"] or task["width"] != protocol["grid_size"]:
            raise ValueError(f"{task_id}: grid size does not match protocol")

    protocol_hash = hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()
    return {
        "protocol": protocol,
        "tasks": tasks,
        "panels": study["panels"],
        "task_ids": task_ids,
        "protocol_hash": protocol_hash,
    }
