"""STUDY-001 runner: learning curves and an A/B/C success matrix over fixed map panels.

Evaluations are read-only. Validation panels drive curves and best-by-validation
selection only; the matrix and retention use the real end-of-stage model
(``after_{key}.pt``), never a reloaded best. The final panel is evaluated only on
explicit request after the full A,B,C sequence. One execution at a time; a failure
stays FAILED with its reason and an ``interrupted.pt`` (no retries, no resume).
"""
from __future__ import annotations  # noqa: I001 -- runtime defaults load first

from . import runtime  # noqa: F401 -- CPU JAX/offline defaults precede scientific imports.

import fcntl
import hashlib
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .checkpoints import build_policy, save_checkpoint
from .envs import make_env
from .evaluation import evaluate_episode
from .learning import A2CTrainer, parameter_digest, seed_everything
from .resources import ResourceGuard
from .runner import snapshot_sources, write_json
from .tracking import Tracker

CLAIM = "STUDY-001 pilot; no superiority claim"
SEQUENCE = ["A", "B", "C"]
TRAIN_FIELDS = ("step", "update", "loss", "policy_loss", "value_loss", "entropy",
                "gradient_norm", "episode_reward", "episode_success", "rollout_reward_mean")
EPSILON = 1e-9  # success rates are k/n; avoid 0.55 + 0.2 > 0.75 float artefacts


def acquisition_step(curve, initial, random, threshold, consecutive, min_gain):
    """Step of the evaluation completing ``consecutive`` qualifying points; never backdated."""
    if consecutive < 1:
        raise ValueError("consecutive must be at least 1")
    floor = max(threshold, initial + min_gain, random + min_gain) - EPSILON
    run = 0
    for point in curve:
        run = run + 1 if point["success_rate"] >= floor else 0
        if run >= consecutive:
            return point["step"]
    return None


def normalized_auc(curve):
    """Trapezoid area of success over steps divided by the step span."""
    if len(curve) < 2:
        return 0.0
    span = curve[-1]["step"] - curve[0]["step"]
    if span <= 0:
        return 0.0
    area = sum((right["step"] - left["step"]) * (left["success_rate"] + right["success_rate"]) / 2
               for left, right in zip(curve, curve[1:]))
    return float(area / span)


def panel_eval(model, task, seeds, sampling_seeds=None, architecture_id="unknown"):
    """Stochastic evaluation of one episode per map seed; model None is the random agent."""
    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("Panel evaluation requires at least one map seed")
    sampling = [None] * len(seeds) if sampling_seeds is None else [int(s) for s in sampling_seeds]
    if len(sampling) != len(seeds):
        raise ValueError("Sampling seeds must pair one-to-one with map seeds")
    records = [evaluate_episode(model, task, seed, architecture_id=architecture_id,
                                sampling_seed=sample)
               for seed, sample in zip(seeds, sampling)]
    result = {"task_id": task["id"],
              "success_rate": float(np.mean([r["success"] for r in records])),
              "reward_mean": float(np.mean([r["reward"] for r in records])),
              "episodes": len(records),
              "environment_steps": sum(len(r["steps"]) for r in records), "seeds": seeds}
    if sampling_seeds is not None:
        result["sampling_seeds"] = sampling
    return result


def _compact(result):
    return {key: result[key] for key in ("success_rate", "reward_mean", "episodes",
                                         "environment_steps")}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _phase(task_keys, single_map):
    if single_map:
        return "single_map"
    return "pilot" if len(task_keys) == 1 else "sequence"


def retention_summary(matrix, task_keys, acquisition):
    """S(A,A) vs S(A,last) on the test panel, from end-of-stage models only."""
    if "A" not in task_keys or task_keys.index("A") == len(task_keys) - 1:
        return {}
    last = task_keys[-1]
    final_row = matrix[f"after_{last}"]["test"]
    before = matrix["after_A"]["test"]["A"]["success_rate"]
    after = final_row["A"]["success_rate"]
    acquired = acquisition["A"]["step"] is not None
    return {"task_key": "A", "panel": "test", "after_stage": last,
            "S_A_A": before, "S_A_last": after, "change_pp": 100 * (after - before),
            "forgetting": max(0.0, before - after), "forgetting_pp": 100 * max(0.0, before - after),
            "A_acquired": acquired, "interpretable": acquired,
            "backward_transfer_pp": {
                key: 100 * (final_row[key]["success_rate"]
                            - matrix[f"after_{key}"]["test"][key]["success_rate"])
                for key in task_keys[:-1]},
            "note": "If A was not acquired, retention is not interpretable; zero forgetting "
                    "alone is not retention evidence."}


def run_study(study, architecture, init_seed, task_keys, *, steps, eval_every, output, root,
              single_map=False, final_evaluation=False, tracking=None):
    protocol = study["protocol"]
    task_keys, init_seed = list(task_keys), int(init_seed)
    steps, eval_every = int(steps), int(eval_every)
    if not task_keys or len(set(task_keys)) != len(task_keys) or any(
            key not in study["task_ids"] for key in task_keys):
        raise ValueError(f"Task keys must be distinct study tasks {sorted(study['task_ids'])}")
    if steps < 0 or eval_every < 1:
        raise ValueError("steps must be nonnegative and eval_every positive")
    if single_map and len(task_keys) != 1:
        raise ValueError("The single-map check trains exactly one task")
    if final_evaluation and task_keys != SEQUENCE:
        raise ValueError("The final panel is evaluated only after the full A,B,C sequence")
    if protocol["algorithm"].get("name", "a2c") != "a2c":
        raise ValueError("Only implemented algorithm a2c is available")
    if set(architecture.get("algorithm", {})) - {"lr"}:
        raise ValueError("An architecture may only override the algorithm's lr")
    phase = _phase(task_keys, single_map)
    architecture_id = architecture["architecture_id"]
    root = Path(root).resolve()
    base = root / "runs" / "study_001"
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = base / f"{stamp}-{phase}-{architecture_id}-s{init_seed}-{''.join(task_keys)}"
    output = Path(output).resolve()
    if output == base or not output.is_relative_to(base):
        raise ValueError("Study output must be a new folder inside runs/study_001/")
    if output.exists():
        raise FileExistsError(f"Study output already exists: {output}")
    base.mkdir(parents=True, exist_ok=True)
    lock = (root / "runs/.execution.lock").open("a")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another laboratory execution is active; models run sequentially") from exc
        output.mkdir(parents=True, exist_ok=False)
        return _execute(study, architecture, init_seed, task_keys, steps, eval_every, output, root,
                        single_map, final_evaluation, tracking, phase)
    finally:
        lock.close()


def _execute(study, architecture, init_seed, task_keys, steps, eval_every, output, root,
             single_map, final_evaluation, tracking, phase):
    protocol = study["protocol"]
    architecture_id = architecture["architecture_id"]
    started = time.perf_counter()
    all_keys = sorted(study["task_ids"])
    tasks = {key: study["tasks"][study["task_ids"][key]] for key in all_keys}
    panels = {key: study["panels"][study["task_ids"][key]] for key in all_keys}
    interactions = {"training": 0, "evaluation": 0, "random_baseline": 0}
    timing = {"training_seconds": 0.0, "evaluation_seconds": 0.0}
    checkpoints, baselines, curves, acquisition, auc, matrix, stages = {}, {}, {}, {}, {}, {}, []
    evaluation_checkpoints = {}
    summary = {"status": "RUNNING", "study_id": protocol["study_id"],
               "protocol_hash": study["protocol_hash"], "phase": phase,
               "architecture_id": architecture_id, "architecture": architecture["architecture"],
               "init_seed": init_seed, "task_keys": task_keys,
               "task_ids": {key: study["task_ids"][key] for key in task_keys},
               "steps": steps, "eval_every": eval_every, "final_evaluation": final_evaluation,
               "algorithm": {**protocol["algorithm"], **architecture.get("algorithm", {})},
               "claim": CLAIM, "output": str(output), "random_baselines": baselines,
               "curves": curves, "acquisition": acquisition, "auc": auc, "matrix": matrix,
               "stages": stages, "checkpoints": checkpoints,
               "evaluation_checkpoints": evaluation_checkpoints, "interactions": interactions,
               "timing": timing}
    single = protocol["single_map"]
    acquisition_rule = {key: protocol["acquisition"][key]
                        for key in ("threshold", "consecutive", "min_gain")}
    guard = ResourceGuard(protocol.get("resources"), output)
    tracker = model = trainer = None

    def measure(key, policy, seeds, sampling=None):
        before = time.perf_counter()
        result = panel_eval(policy, tasks[key], seeds, sampling,
                            architecture_id=architecture_id if policy is not None else "random")
        timing["evaluation_seconds"] += time.perf_counter() - before
        bucket = "evaluation" if policy is not None else "random_baseline"
        interactions[bucket] += result["environment_steps"]
        return result

    def validation(key, policy):
        if single_map:
            episodes = int(single["episodes"])
            return measure(key, policy, [panels[key]["train"][0]] * episodes, range(episodes))
        return measure(key, policy, panels[key]["validation"])

    def checkpoint(name, metadata, registry=checkpoints, folder="checkpoints"):
        path = output / folder / f"{name}.pt"
        save_checkpoint(path, model, trainer, metadata={
            "study_id": protocol["study_id"], "protocol_hash": study["protocol_hash"],
            "phase": phase, "architecture_id": architecture_id, "init_seed": init_seed,
            "total_steps": trainer.total_steps if trainer else 0, **metadata})
        registry[name] = {"path": str(path.relative_to(output)), "sha256": _sha256(path)}
        return registry[name]

    def matrix_row(name):
        row = {"checkpoint": checkpoints[name]["path"], "sha256": checkpoints[name]["sha256"],
               "total_step": trainer.total_steps, "test": {}, "known": {}}
        for key in all_keys:
            row["test"][key] = _compact(measure(key, model, panels[key]["test"]))
            row["known"][key] = _compact(measure(key, model, panels[key]["known"]))
        matrix[name] = row
        tracker.metrics({f"{key}.{panel}_success": row[panel][key]["success_rate"]
                         for key in all_keys for panel in ("test", "known")},
                        len(matrix) - 1, f"matrix.{name}.")

    def record_point(key, stage_step, curve, best, curve_log):
        result = validation(key, model)
        # Every evaluation keeps its weights so the curve can be re-evaluated later (C08, R = 3).
        saved = checkpoint(f"{key}_{stage_step:07d}", {
            "role": "evaluation", "task_key": key, "task_id": tasks[key]["id"],
            "stage_step": stage_step, "validation_success": result["success_rate"]},
            evaluation_checkpoints, "checkpoints/evaluations")
        point = {"task_key": key, "step": stage_step, "total_step": trainer.total_steps,
                 **_compact(result), "checkpoint": saved["path"]}
        curve.append(point)
        curve_log.write(json.dumps(point, allow_nan=False) + "\n")
        curve_log.flush()
        tracker.metrics({"validation_success": point["success_rate"],
                         "validation_reward": point["reward_mean"]},
                        trainer.total_steps, f"curve.{key}.")
        if best.get("success_rate") is None or point["success_rate"] > best["success_rate"]:
            best.update(success_rate=point["success_rate"], step=stage_step)
            checkpoint(f"best_{key}", {"role": "best_by_validation", "selection_only": True,
                                       "task_key": key, "task_id": tasks[key]["id"],
                                       "stage_step": stage_step,
                                       "validation_success": point["success_rate"]})
        write_json(output / "progress.json", {**summary, "stage": key, "stage_step": stage_step})

    try:
        guard.check()
        write_json(output / "study.json", {
            "protocol": protocol, "protocol_hash": study["protocol_hash"],
            "task_ids": study["task_ids"], "tasks": study["tasks"], "panels": study["panels"],
            "architecture": architecture})
        sources = snapshot_sources(output)
        write_json(output / "source_identity.json", sources)
        summary["source_identity"] = sources["identity"]
        summary["versions"] = {name: importlib.metadata.version(name)
                               for name in ("torch", "numpy", "xminigrid", "jax", "mlflow-skinny")}
        tracker = Tracker(root, tracking or protocol.get("tracking", {}),
                          f"STUDY-001 {phase} {architecture_id} seed {init_seed} "
                          f"{'-'.join(task_keys)}")
        summary["mlflow"] = tracker.reference()
        for tag, value in {"study_id": protocol["study_id"], "phase": phase,
                           "architecture": architecture_id, "init_seed": init_seed,
                           "protocol_hash": study["protocol_hash"], "claim": CLAIM,
                           "task_keys": ",".join(task_keys)}.items():
            tracker.client.set_tag(tracker.run_id, tag, str(value))

        first = task_keys[0]
        pools = {key: [panels[key]["train"][0]] if single_map else list(panels[key]["train"])
                 for key in task_keys}
        env = make_env(tasks[first])
        seed_everything(init_seed)
        model = build_policy(architecture["architecture"], env.observation_dim, env.action_count,
                             "cpu")
        algorithm = {k: v for k, v in summary["algorithm"].items() if k != "name"}
        trainer = A2CTrainer(model, env, pools[first][0] + init_seed, algorithm,
                             seed_pool=pools[first])
        summary["parameters"] = model.parameter_counts()
        summary["initial_trainable_sha256"] = parameter_digest(model, frozen=False)
        tracker.params({"study_id": protocol["study_id"], "phase": phase,
                        "architecture_id": architecture_id,
                        "architecture": architecture["architecture"], "init_seed": init_seed,
                        "task_keys": task_keys, "task_ids": summary["task_ids"], "steps": steps,
                        "eval_every": eval_every, "protocol_hash": study["protocol_hash"],
                        "algorithm": summary["algorithm"], "source_identity": sources["identity"],
                        "trainable_parameters": summary["parameters"]["trainable"]})

        for key in task_keys if single_map else all_keys:
            if single_map:
                episodes = int(single["episodes"])
                baselines[key] = {"single_map": _compact(measure(
                    key, None, [panels[key]["train"][0]] * episodes, range(episodes)))}
            else:
                baselines[key] = {panel: _compact(measure(key, None, panels[key][panel]))
                                  for panel in ("validation", "test")}
        checkpoint("initial", {"role": "initial", "task_key": first, "task_id": tasks[first]["id"]})
        if not single_map:
            matrix_row("initial")

        check_every = int(protocol.get("resource_check_every_updates", 25))
        since_check = 0
        with (output / "train_metrics.jsonl").open("x") as train_log, \
                (output / "curve.jsonl").open("x") as curve_log:
            for index, key in enumerate(task_keys):
                guard.check()
                if index:
                    trainer.set_task(make_env(tasks[key]), pools[key][0] + init_seed,
                                     seed_pool=pools[key])
                if (trainer.env.observation_dim, trainer.env.action_count) != (
                        model.input_dim, model.action_count):
                    raise ValueError(f"Task {key} differs in observation or action dimensions")
                stage_start, curve, best = trainer.total_steps, [], {}
                curves[key] = curve
                record_point(key, 0, curve, best, curve_log)
                stage_step, next_eval = 0, eval_every
                while stage_step < steps:
                    count = min(trainer.config["rollout_steps"], steps - stage_step)
                    before = time.perf_counter()
                    row = trainer.update(count)
                    timing["training_seconds"] += time.perf_counter() - before
                    stage_step += count
                    since_check += 1
                    if since_check >= check_every:
                        guard.check()
                        since_check = 0
                    entry = {"task_key": key, "stage_step": stage_step,
                             **{field: row[field] for field in TRAIN_FIELDS},
                             "episodes_completed": len(row["completed_episodes"])}
                    train_log.write(json.dumps(entry, allow_nan=False) + "\n")
                    if stage_step >= next_eval or stage_step == steps:
                        train_log.flush()
                        record_point(key, stage_step, curve, best, curve_log)
                        while next_eval <= stage_step:
                            next_eval += eval_every
                checkpoint(f"after_{key}", {"role": "end_of_stage", "task_key": key,
                                            "task_id": tasks[key]["id"], "stage_step": stage_step})
                if not single_map:
                    matrix_row(f"after_{key}")
                random_success = baselines[key]["single_map" if single_map else "validation"][
                    "success_rate"]
                step = acquisition_step(curve, curve[0]["success_rate"], random_success,
                                        **acquisition_rule)
                acquisition[key] = {"step": step, "acquired": step is not None,
                                    "initial_success": curve[0]["success_rate"],
                                    "random_success": random_success, "rule": acquisition_rule}
                auc[key] = normalized_auc(curve)
                stages.append({"task_key": key, "task_id": tasks[key]["id"],
                               "trainer_seed": trainer.seed, "seed_pool_size": len(pools[key]),
                               "start_total_step": stage_start,
                               "end_total_step": trainer.total_steps,
                               "best_validation": dict(best)})
                stage_metrics = {"auc": auc[key]}
                if step is not None:
                    stage_metrics["acquisition_step"] = step
                tracker.metrics(stage_metrics, trainer.total_steps, f"stage.{key}.")

        if single_map:
            curve = curves[task_keys[0]]
            summary["single_map_check"] = {
                "threshold": single["threshold"], "final_success": curve[-1]["success_rate"],
                "best_success": max(point["success_rate"] for point in curve),
                "passed": curve[-1]["success_rate"] >= single["threshold"] - EPSILON,
                "note": "Circuit check on one map; not a study result."}
        else:
            summary["retention"] = retention_summary(matrix, task_keys, acquisition)
            if summary["retention"]:
                tracker.metrics({k: v for k, v in summary["retention"].items()
                                 if k in ("S_A_A", "S_A_last", "change_pp", "forgetting_pp")},
                                trainer.total_steps, "retention.")
        if final_evaluation:
            summary["final_panel"] = {
                "checkpoint": checkpoints["after_C"],
                "results": {key: _compact(measure(key, model, panels[key]["final"]))
                            for key in all_keys}}
        final_digest = parameter_digest(model, frozen=False)
        if trainer.total_steps > 0 and final_digest == summary["initial_trainable_sha256"]:
            raise RuntimeError("Training did not change trainable parameters")
        interactions["training"] = trainer.total_steps
        summary.update(status="FINISHED", final_trainable_sha256=final_digest,
                       elapsed_seconds=time.perf_counter() - started)
        write_json(output / "resources.json", guard.samples)
        write_json(output / "summary.json", summary)
        write_json(output / "progress.json", summary)
        tracker.metrics({"training_interactions": interactions["training"],
                         "evaluation_interactions": interactions["evaluation"],
                         "elapsed_seconds": summary["elapsed_seconds"]}, trainer.total_steps)
        tracker.finish(output)
        trainer.env.close()
        return summary
    except Exception as exc:
        if trainer is not None:
            interactions["training"] = trainer.total_steps
        summary.update(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                       elapsed_seconds=time.perf_counter() - started)
        if model is not None:
            try:
                checkpoint("interrupted", {"role": "interrupted", "interrupted": True})
            except Exception as save_error:
                summary["checkpoint_error"] = f"{type(save_error).__name__}: {save_error}"
        write_json(output / "resources.json", guard.samples)
        write_json(output / "summary.json", summary)
        write_json(output / "progress.json", summary)
        if tracker is not None:
            try:
                tracker.finish(output, "FAILED")
            except Exception as tracking_error:
                summary["tracking_error"] = f"{type(tracking_error).__name__}: {tracking_error}"
                write_json(output / "summary.json", summary)
        raise
