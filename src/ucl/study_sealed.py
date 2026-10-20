"""A sealed panel of new maps, built without looking at any model and opened once (C01).

``build_sealed_panel`` walks map seeds from a fixed base and keeps the first ``size`` maps whose
physical identity (grid, agent position, direction, pocket) appears in no panel of the study, so
the sealed maps are new to every run. It writes ``configs/sealed/<name>.json`` and refuses to
overwrite it.

``open_sealed`` plays one chosen checkpoint of every run that an analysis spec's plans list on
that panel, ``replicas`` times per map with sampling seeds ``map seed + r * 10,000,019`` for
r = 1..replicas, with weights fixed. Opening is recorded in ``runs/sealed/<evaluation id>/
opening.json`` together with the spec's hash; a second opening with a different spec is refused.
If the process stops part way, running it again finishes the runs that are missing and never
re-plays a finished one.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_DIR = "sealed"


def build_sealed_panel(root, study_dir: str, task_key: str, base: int, size: int,
                       name: str) -> dict:
    from .envs import make_env
    from .study_panels import IDENTITY_DEFINITION, load_study, physical_identity

    root = Path(root).resolve()
    target = root / "configs" / "sealed" / f"{name}.json"
    if target.exists():
        raise FileExistsError(f"Sealed panel already exists: {target}")
    study = load_study(root / study_dir)
    task_id = study["task_ids"][task_key]
    panels = json.loads((root / study_dir / "panels.json").read_text())
    used_seeds = {seed for panel in study["panels"][task_id].values() for seed in panel}
    used = set(panels["identities"][task_id].values())
    env = make_env(study["tasks"][task_id])
    seeds, identities, scanned = [], {}, 0
    try:
        while len(seeds) < size:
            if scanned >= 50 * size:
                raise ValueError(f"Only {len(seeds)} new maps in {scanned} seeds")
            seed = base + scanned
            scanned += 1
            if seed in used_seeds:
                continue
            env.reset(seed)
            identity = physical_identity(env)
            if identity in used:
                continue
            used.add(identity)
            seeds.append(seed)
            identities[str(seed)] = identity
    finally:
        env.close()
    panel = {"kind": "sealed_panel", "name": name, "study_dir": study_dir,
             "protocol_hash": study["protocol_hash"], "task_key": task_key, "task_id": task_id,
             "base": base, "scanned": scanned, "seeds": seeds, "identities": identities,
             "excluded_maps": len(set(panels["identities"][task_id].values())),
             "identity_definition": IDENTITY_DEFINITION,
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(panel, indent=1))
    return panel


def result_path(root, evaluation_id: str, run_id: str) -> Path:
    return Path(root).resolve() / "runs" / OUTPUT_DIR / evaluation_id / run_id / "sealed.json"


def open_sealed(root, spec_path) -> dict:
    """Evaluate every run of the spec's plans on its sealed panel; returns a short status."""
    from .checkpoints import load_checkpoint
    from .evaluation import evaluate_episode
    from .study_chain import load_plan, plan_items
    from .study_format import resolve_study_run
    from .study_panels import load_study
    from .study_reevaluation import _checked_checkpoint, sampling_seed

    root = Path(root).resolve()
    spec_path = Path(spec_path) if Path(spec_path).is_absolute() else root / spec_path
    text = spec_path.read_text()
    spec = json.loads(text)
    sealed = spec["sealed"]
    panel = json.loads((root / sealed["panel"]).read_text())
    study = load_study(root / panel["study_dir"])
    if study["protocol_hash"] != panel["protocol_hash"]:
        raise ValueError("The study changed after the sealed panel was built")
    task = study["tasks"][panel["task_id"]]
    replicas, checkpoint = int(sealed["replicas"]), sealed["checkpoint"]
    folder = root / "runs" / OUTPUT_DIR / sealed["evaluation_id"]
    opening = folder / "opening.json"
    spec_hash = hashlib.sha256(text.encode()).hexdigest()
    if opening.is_file() and json.loads(opening.read_text())["spec_sha256"] != spec_hash:
        raise ValueError("This sealed panel was already opened with another analysis spec")
    runs = [(load_plan(root / path), item) for path in spec["plans"]
            for item in plan_items(load_plan(root / path))]
    lock = (root / "runs/.execution.lock").open("a")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another laboratory execution is active") from exc
        if not opening.is_file():
            folder.mkdir(parents=True, exist_ok=True)
            opening.write_text(json.dumps({
                "spec": str(spec_path.relative_to(root)), "spec_sha256": spec_hash,
                "panel": sealed["panel"], "checkpoint": checkpoint, "replicas": replicas,
                "runs": len(runs),
                "opened": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=1))
        done = 0
        for plan, item in runs:
            run_id = (root / plan["output_dir"] / item["name"]).relative_to(root / "runs")
            run_id = run_id.as_posix()
            target = result_path(root, sealed["evaluation_id"], run_id)
            if target.is_file():
                done += 1
                continue
            directory, summary = resolve_study_run(root, run_id)
            finished = summary.get("status") == "FINISHED"
            if not finished or summary["protocol_hash"] != panel["protocol_hash"]:
                raise ValueError(f"{run_id} is not a finished run of the panel's study")
            path = _checked_checkpoint(directory, summary["checkpoints"][checkpoint])
            model, _, _ = load_checkpoint(path, expected_architecture=summary["architecture"])
            started = time.perf_counter()
            rows = []
            for replica in range(1, replicas + 1):
                success = [int(bool(evaluate_episode(
                    model, task, seed, architecture_id=summary["architecture_id"],
                    sampling_seed=sampling_seed(seed, replica))["success"]))
                    for seed in panel["seeds"]]
                rows.append({"replica": replica, "success": success,
                             "success_rate": sum(success) / len(success)})
            episodes = replicas * len(panel["seeds"])
            result = {"kind": "sealed_evaluation", "evaluation_id": sealed["evaluation_id"],
                      "run": run_id, "label": item["label"], "seed": item["seed"],
                      "checkpoint": checkpoint,
                      "checkpoint_sha256": summary["checkpoints"][checkpoint]["sha256"],
                      "panel": sealed["panel"], "replicas": rows,
                      "success_rate": sum(sum(r["success"]) for r in rows) / episodes,
                      "episodes": episodes, "seconds": time.perf_counter() - started}
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=1))
            done += 1
        return {"evaluation_id": sealed["evaluation_id"], "runs": len(runs), "evaluated": done}
    finally:
        lock.close()
