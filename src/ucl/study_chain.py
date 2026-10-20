"""Run a preregistered list of study runs as a chain of events (C08 Part I, C01 Part II).

A plan (``plans/<name>.json``) fixes everything before any result exists: the study, the
architectures, the tasks, the budget and the seeds. Seeds come from a fixed rule
(``plan_seeds``): the pilot seeds first, then new ones drawn from NumPy's ``SeedSequence`` with
the plan's master seed (skipping ``exclude_seeds``), so anyone can rebuild the same list.

An architecture entry is a config path, or an object with ``config``, an optional ``label``
(the run-name prefix), ``algorithm`` overrides (only ``lr``), ``seed_count`` (this arm uses the
first n seeds, so arms stay paired by seed) and ``lr_from`` (take the learning rate that the
plan's ``after`` plan selected for that group). With ``shuffle_within_seed`` the order of the
arms inside each seed block is a fixed permutation drawn from the master seed.

Each event starts the next one: «run finished» starts its re-evaluation, «re-evaluation finished»
starts the next run, and a finished ``after`` plan starts this one, after writing which learning
rates its preregistered rule selected (``plan_resolved``; never recomputed). The chain's own
journal (``runs/chains/<plan id>/journal.jsonl``) records every event; nothing scans ``runs/``
to decide what to do. If a step fails or the process is killed, the chain stops and shows that
item as FAILED or INTERRUPTED with its reason. Running the command again with ``--retry``
repeats that item with the same seed; there are no silent retries.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MAX_SEED = 1_000_000
EVENTS = ("plan_resolved", "run_started", "run_finished", "reevaluation_finished", "failed")
ALGORITHM_OVERRIDES = {"lr"}
SELECTION_RULE = "highest_mean_validation_auc"


def plan_seeds(plan: dict) -> list[int]:
    """Pilot seeds first, then distinct new seeds from SeedSequence(master_seed)."""
    seeds = [int(seed) for seed in plan.get("pilot_seeds", [])]
    excluded = {int(seed) for seed in plan.get("exclude_seeds", [])}
    count = int(plan["seed_count"])
    if len(set(seeds)) != len(seeds) or count < len(seeds) or excluded & set(seeds):
        raise ValueError("Pilot seeds must be distinct, not excluded and no more than seed_count")
    sequence = np.random.SeedSequence(int(plan["master_seed"]))
    while len(seeds) < count:
        child = sequence.spawn(1)[0]
        seed = int(child.generate_state(1, dtype=np.uint32)[0]) % MAX_SEED
        if seed not in seeds and seed not in excluded:
            seeds.append(seed)
    return seeds


def plan_hash(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def plan_architectures(plan: dict) -> list[dict]:
    """Architecture entries in plan order, each with config, label, algorithm and seed_count."""
    arms = []
    for entry in plan["architectures"]:
        entry = {"config": entry} if isinstance(entry, str) else dict(entry)
        unknown = set(entry) - {"config", "label", "algorithm", "seed_count", "lr_from"}
        if unknown:
            raise ValueError(f"Unknown architecture entry fields {sorted(unknown)}")
        algorithm = entry.get("algorithm", {})
        if set(algorithm) - ALGORITHM_OVERRIDES:
            raise ValueError(f"Plans may only override {sorted(ALGORITHM_OVERRIDES)}")
        if "lr_from" in entry and ("lr" in algorithm or "after" not in plan):
            raise ValueError("lr_from needs an 'after' plan and no fixed lr")
        default_label = Path(entry["config"]).stem.removeprefix("arch-")
        arms.append({"config": entry["config"], "label": entry.get("label") or default_label,
                     "algorithm": algorithm, "lr_from": entry.get("lr_from"),
                     "seed_count": int(entry.get("seed_count", plan["seed_count"]))})
    labels = [arm["label"] for arm in arms]
    if len(set(labels)) != len(labels) or any("-s" in label for label in labels):
        raise ValueError("Architecture labels must be distinct and must not contain '-s'")
    if any(not 1 <= arm["seed_count"] <= int(plan["seed_count"]) for arm in arms):
        raise ValueError("An architecture's seed_count must be between 1 and the plan's")
    return arms


def plan_items(plan: dict) -> list[dict]:
    """Every run of the plan in its fixed order: seed by seed, then the arms of that seed."""
    arms = plan_architectures(plan)
    items = []
    for position, seed in enumerate(plan_seeds(plan)):
        block = [arm for arm in arms if position < arm["seed_count"]]
        if plan.get("shuffle_within_seed"):
            order = np.random.default_rng([int(plan["master_seed"]), seed]).permutation(len(block))
            block = [block[i] for i in order]
        for arm in block:
            items.append({"index": len(items), "seed": seed, "architecture": arm["config"],
                          "label": arm["label"], "algorithm": arm["algorithm"],
                          "lr_from": arm["lr_from"],
                          "name": f"{arm['label']}-s{seed}-{''.join(plan['tasks'])}"})
    return items


def load_plan(path) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"plan_id", "study_dir", "architectures", "tasks", "steps", "eval_every",
                "replicas", "master_seed", "seed_count", "output_dir"}
    missing = required - set(plan)
    if missing:
        raise ValueError(f"Plan is missing {sorted(missing)}")
    if not str(plan["output_dir"]).startswith("runs/study_001/"):
        raise ValueError("Plan output_dir must be inside runs/study_001/")
    if "after" in plan and plan.get("selection", {}).get("rule") != SELECTION_RULE:
        raise ValueError(f"A plan with 'after' must declare selection.rule = {SELECTION_RULE}")
    plan_items(plan)  # validates architectures and seeds
    return plan


class Journal:
    """Append-only event log of one chain; its last event per item is that item's state."""

    def __init__(self, root: Path, plan: dict):
        self.path = Path(root) / "runs" / "chains" / plan["plan_id"] / "journal.jsonl"
        self.plan_hash = plan_hash(plan)

    def events(self) -> list[dict]:
        if not self.path.is_file():
            return []
        events = [json.loads(line) for line in self.path.read_text().splitlines() if line]
        if events and events[0].get("plan_hash") != self.plan_hash:
            raise ValueError("The plan changed after this chain started; start a new plan_id")
        return events

    def write(self, event: str, **fields) -> dict:
        if event not in EVENTS:
            raise ValueError(f"Unknown chain event {event}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"event": event, "plan_hash": self.plan_hash,
                 "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(entry, allow_nan=False) + "\n")
        return entry

    def resolution(self) -> dict | None:
        return next((e for e in self.events() if e["event"] == "plan_resolved"), None)

    def states(self) -> dict[int, dict]:
        """Per item: its last event, and whether its run finished since it was last started."""
        states = {}
        for entry in self.events():
            if "item" not in entry:
                continue
            state = states.setdefault(entry["item"], {"run_done": False})
            if entry["event"] == "run_started":
                state["run_done"] = False
            elif entry["event"] == "run_finished":
                state["run_done"] = True
            state["last"] = entry
        return states


def _after_plan(root, plan: dict) -> dict | None:
    if "after" not in plan:
        return None
    path = Path(plan["after"])
    return load_plan(path if path.is_absolute() else Path(root) / path)


def chain_status(root, plan: dict) -> dict:
    """Where the chain stands, read only from its own journal (and its ``after`` plan's)."""
    items = plan_items(plan)
    journal = Journal(root, plan)
    states = journal.states()
    last = {index: state["last"]["event"] for index, state in states.items()}
    done = [i for i in items if last.get(i["index"]) == "reevaluation_finished"]
    blocked = None
    for item in items:
        event = last.get(item["index"])
        if event is None or event == "reevaluation_finished":
            continue
        reasons = {"failed": states[item["index"]]["last"].get("error"),
                   "run_started": "the process stopped while this run was training",
                   "run_finished": "the process stopped before its re-evaluation finished"}
        blocked = {"item": item, "status": "FAILED" if event == "failed" else "INTERRUPTED",
                   "reason": reasons[event]}
        break
    status = {"plan_id": plan["plan_id"], "total": len(items), "finished": len(done),
              "blocked": blocked}
    after = _after_plan(root, plan)
    if after is not None:
        resolution = journal.resolution()
        status["after"] = chain_status(root, after)
        status["learning_rates"] = resolution["learning_rates"] if resolution else None
    return status


def select_learning_rates(root, after: dict) -> dict:
    """The preregistered rule: per label group of the ``after`` plan, the learning rate with the
    highest mean normalized validation AUC over its runs; ties go to the one listed first."""
    root = Path(root).resolve()
    task = after["tasks"][0]
    scores = {}
    for item in plan_items(after):
        if "lr" not in item["algorithm"]:
            raise ValueError(f"{item['name']}: every arm of a learning-rate plan needs an lr")
        group = item["label"].rsplit("-lr", 1)[0]
        summary = json.loads((root / after["output_dir"] / item["name"] / "summary.json")
                             .read_text())
        scores.setdefault(group, {}).setdefault(item["algorithm"]["lr"], []).append(
            summary["auc"][task])
    evidence = {group: {str(lr): float(np.mean(values)) for lr, values in by_lr.items()}
                for group, by_lr in scores.items()}
    chosen = {group: max(by_lr, key=lambda lr: (np.mean(by_lr[lr]), -list(by_lr).index(lr)))
              for group, by_lr in scores.items()}
    return {"learning_rates": chosen, "evidence": evidence, "rule": SELECTION_RULE}


def run_chain(root, plan: dict, *, retry: bool = False, limit: int | None = None,
              study=None, run_study_fn=None, reevaluate_fn=None) -> dict:
    """Advance the chain event by event until it ends, fails, or ``limit`` items are completed."""
    from .cli import read_config
    from .study import run_study
    from .study_panels import load_study
    from .study_reevaluation import reevaluate_run

    run_study_fn = run_study_fn or run_study
    reevaluate_fn = reevaluate_fn or reevaluate_run
    root = Path(root).resolve()
    journal = Journal(root, plan)
    after = _after_plan(root, plan)
    if after is not None and journal.resolution() is None:
        done = run_chain(root, after, retry=retry, limit=limit, study=study,
                         run_study_fn=run_study_fn, reevaluate_fn=reevaluate_fn)
        if done["finished"] != done["total"]:
            return chain_status(root, plan)
        journal.write("plan_resolved", after=after["plan_id"], **select_learning_rates(root, after))
        retry = False  # the retry was for the after plan
    status = chain_status(root, plan)
    if status["blocked"] and not retry:
        blocked = status["blocked"]
        raise RuntimeError(f"Item {blocked['item']['index']} ({blocked['item']['name']}) is "
                           f"{blocked['status']}: {blocked['reason']}. Run again with --retry.")
    resolution = journal.resolution()
    study = study if study is not None else load_study(root / plan["study_dir"])
    states = journal.states()
    completed = 0
    for item in plan_items(plan):
        state = states.get(item["index"], {"run_done": False})
        if state.get("last", {}).get("event") == "reevaluation_finished":
            continue
        if limit is not None and completed >= limit:
            break
        output = root / plan["output_dir"] / item["name"]
        run_id = output.relative_to(root / "runs").as_posix()
        try:
            if not state["run_done"]:
                if output.exists():
                    # A retried run keeps its failed attempt as evidence, next to the plan output.
                    attempts = root / plan["output_dir"] / "_failed"
                    attempts.mkdir(parents=True, exist_ok=True)
                    number = len(list(attempts.glob(f"{item['name']}-*"))) + 1
                    shutil.move(str(output), str(attempts / f"{item['name']}-{number}"))
                algorithm = dict(item["algorithm"])
                if item["lr_from"]:
                    algorithm["lr"] = resolution["learning_rates"][item["lr_from"]]
                architecture = read_config(item["architecture"])
                if algorithm:
                    architecture = {**architecture, "algorithm": algorithm}
                journal.write("run_started", item=item["index"], name=item["name"],
                              seed=item["seed"], run=run_id, algorithm=algorithm)
                run_study_fn(study, architecture, item["seed"], plan["tasks"],
                             steps=plan["steps"], eval_every=plan["eval_every"],
                             output=output, root=root)
                journal.write("run_finished", item=item["index"], run=run_id)
            reevaluate_fn(root, run_id, plan["replicas"])
            journal.write("reevaluation_finished", item=item["index"], run=run_id)
        except Exception as exc:
            journal.write("failed", item=item["index"], run=run_id,
                          error=f"{type(exc).__name__}: {exc}")
            raise
        completed += 1
    return chain_status(root, plan)
