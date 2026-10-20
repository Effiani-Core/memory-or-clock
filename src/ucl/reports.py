"""Descriptive comparisons with explicit identities, exposure and transfer criteria.

Reading reports needs only the standard library; training/tracking imports are lazy.
"""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from .tasks import load_tasks, task_identity

ROOT = Path(__file__).resolve().parents[2]
SUCCESS_THRESHOLD = 0.75
SUSTAINED_OBSERVATIONS = 2


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def new_output(label):
    return ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + label)


def run_experiment(*args, **kwargs):
    from .runner import run_experiment as run
    return run(*args, **kwargs)


def _new_tracker(config):
    from .tracking import Tracker
    return Tracker(ROOT, config, "Learning to learn - smoke comparison")


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _saved_tasks(directory):
    path = Path(directory) / "tasks_manifest.json"
    return json.loads(path.read_text()).get("tasks", {}) if path.exists() else {}


def _task_hash(tasks, identifier):
    if identifier not in tasks:
        return None
    task = tasks[identifier]
    actual = task_identity(task)
    if task.get("task_sha256") != actual:
        raise ValueError(f"Recorded task identity mismatch: {identifier}")
    return actual


def _scientific_config(config, tasks):
    # Replicate seed and bookkeeping are not different scientific protocols.
    result = copy.deepcopy(config)
    for key in ("seed", "source_id", "source_identity", "source_revision", "tracking", "resources",
                "architecture_id", "name", "claim", "task_manifest", "output"):
        result.pop(key, None)
    identifiers = {config.get(key) for key in ("task_id", "evaluation_task", "heldout_task")}
    identifiers.update(stage.get("task_id") for stage in config.get("stages", []))
    result["task_content_hashes"] = {identifier: _task_hash(tasks, identifier)
                                     for identifier in sorted(value for value in identifiers if value)}
    return result


def comparison_report(directories, output):
    output = Path(output)
    if not output.is_absolute():
        output = ROOT / output
    if not output.resolve().is_relative_to(ROOT / "runs"):
        raise ValueError("Reports must stay within runs/")
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for directory in directories:
        path = Path(directory)
        if not path.is_absolute():
            path = ROOT / path
        summary = json.loads((path / "summary.json").read_text())
        if summary["status"] != "FINISHED":
            rows.append({"output": str(path), "status": summary["status"],
                         "architecture_id": summary.get("architecture_id"), "error": summary.get("error")})
            continue
        tasks = _saved_tasks(path)
        config, metrics = summary["config"], summary["metrics"]
        scientific = _scientific_config(config, tasks)
        evaluation_hash = _task_hash(tasks, metrics["task_id"])
        rows.append({"output": str(path), "status": "FINISHED", "architecture_id": summary["architecture_id"],
                     "seed": config["seed"], "training_steps": summary["training_steps"],
                     "new_training_steps": summary.get("new_training_steps", summary["training_steps"]),
                     "evaluation_task": metrics["task_id"], "evaluation_task_sha256": evaluation_hash,
                     "evaluation_seeds": metrics["seeds"], "success_rate": metrics["success_rate"],
                     "reward_mean": metrics["reward_mean"], "elapsed_seconds": summary["elapsed_seconds"],
                     "initial_parameters": summary["initial_parameters"], "final_parameters": summary["final_parameters"],
                     "growth_events": summary["growth_events"], "algorithm": config.get("algorithm"),
                     "scientific_configuration": scientific, "scientific_configuration_sha256": _identity(scientific),
                     "source_identity": summary.get("source_identity"),
                     "cohort_verified": evaluation_hash is not None and "algorithm" in config and "architecture" in config})
    grouped = {}
    for row in rows:
        if row["status"] == "FINISHED":
            cohort = (row["architecture_id"], row["training_steps"], row["new_training_steps"],
                      row["evaluation_task"], row["evaluation_task_sha256"], tuple(row["evaluation_seeds"]),
                      row["scientific_configuration_sha256"], _identity(row["initial_parameters"]), _identity(row["final_parameters"]))
            grouped.setdefault(cohort, []).append(row)
    aggregates = []
    for group in grouped.values():
        first = group[0]
        if len({row["seed"] for row in group}) != len(group):
            raise ValueError("Repeated seed within one comparison cohort; select one run per seed")
        values = [row["success_rate"] for row in group]
        aggregates.append({"architecture_id": first["architecture_id"], "training_steps": first["training_steps"],
                           "new_training_steps": first["new_training_steps"], "evaluation_task": first["evaluation_task"],
                           "evaluation_task_sha256": first["evaluation_task_sha256"], "evaluation_seeds": first["evaluation_seeds"],
                           "scientific_configuration_sha256": first["scientific_configuration_sha256"],
                           "seeds": [row["seed"] for row in group], "mean_success_rate": statistics.mean(values),
                           "std_between_seeds": statistics.stdev(values) if len(group) > 1 else None,
                           "cohort_verified": all(row["cohort_verified"] for row in group),
                           "source_identities": sorted({row["source_identity"] for row in group if row["source_identity"]}),
                           "source_identity_is_grouping_key": False})
    comparisons = [row for row in rows if row["status"] == "FINISHED"]
    verified = bool(comparisons) and all(row["cohort_verified"] for row in comparisons)
    equal_tasks = verified and len({(row["evaluation_task_sha256"], tuple(row["evaluation_seeds"])) for row in comparisons}) == 1
    equal_exposure = bool(comparisons) and len({(row["training_steps"], row["new_training_steps"]) for row in comparisons}) == 1
    report = {"rows": rows, "aggregates": aggregates, "same_evaluation_task_and_seeds": equal_tasks,
              "same_training_exposure": equal_exposure, "all_cohort_identities_verified": verified,
              "claim": "Short smoke runs validate infrastructure. Differences are not evidence of superiority or meta-learning.",
              "cohort_note": "Task content, algorithm and scientific configuration separate cohorts. Replicate seed and source identity do not; source identities remain visible for audit. Missing identities prevent verified comparison.",
              "parameter_budget_note": "Initial/final trainable and frozen counts are explicit; use CONTROL-RIM-MAX for exact final-capacity growth control."}
    write_json(output, report)
    return report


def _count(record, key):
    value = record.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"Actual nonnegative {key} is required for exposure accounting")
    return value


def _exposure(result):
    stages = result["stages"]
    source, target, other = 0, 0, 0
    for stage in stages:
        delta = _count(stage, "training_steps_after") - _count(stage, "training_steps_before")
        if delta < 0:
            raise ValueError("Training counters decreased within a stage")
        if stage["purpose"] == "prior_experience":
            source += delta
        elif stage["purpose"] == "target_curve":
            target += delta
        else:
            other += delta
    training = result.get("new_training_steps", result["training_steps"])
    if source + target + other != training:
        raise ValueError("Stage exposure does not match actual new training interactions")
    validation = result.get("evaluations", [])
    final = [result["metrics"]] if result.get("mode", "train") != "evaluate" else []
    records = stages + validation + final
    evaluation_steps = sum(_count(record, "environment_steps") for record in records)
    evaluation_episodes = sum(_count(record, "episodes") for record in records)
    return {"source_training_interactions": source, "target_training_interactions": target,
            "other_training_interactions": other, "total_training_interactions": training,
            "source_evaluation_interactions": sum(_count(s, "environment_steps") for s in stages if s["purpose"] == "prior_experience"),
            "target_curve_evaluation_interactions": sum(_count(s, "environment_steps") for s in stages if s["purpose"] == "target_curve"),
            "validation_interactions": sum(_count(s, "environment_steps") for s in validation),
            "final_evaluation_interactions": sum(_count(s, "environment_steps") for s in final),
            "total_evaluation_interactions": evaluation_steps, "evaluation_episodes": evaluation_episodes,
            "total_environment_interactions": training + evaluation_steps}


def _curve(stages, target, expected_seeds, target_steps):
    if not stages:
        raise ValueError("Target learning curve is missing")
    offset = stages[0]["training_steps_before"]
    points = []
    for stage in stages:
        if stage["task_id"] != target or stage["seeds"] != expected_seeds:
            raise ValueError("Target task or evaluation seeds differ from the matched protocol")
        if stage["episodes"] != len(expected_seeds):
            raise ValueError("Target evaluation episode count differs from seed count")
        score = float(stage["success_rate"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Success rate must be finite in [0, 1]")
        points.append({"target_interactions": stage["training_steps_after"] - offset,
                       "success_rate": score, "reward_mean": stage["reward_mean"],
                       "evaluation_seeds": list(stage["seeds"]), "evaluation_interactions": _count(stage, "environment_steps")})
    positions = [point["target_interactions"] for point in points]
    if positions[0] != 0 or positions[-1] != target_steps or any(b <= a for a, b in zip(positions, positions[1:])):
        raise ValueError("Target curve must start at zero, increase, and end at actual requested target exposure")
    auc = sum((b["target_interactions"] - a["target_interactions"]) * (a["success_rate"] + b["success_rate"]) / 2
              for a, b in zip(points, points[1:])) / target_steps
    consecutive = 0
    crossing = None
    for point in points:
        consecutive = consecutive + 1 if point["success_rate"] >= SUCCESS_THRESHOLD else 0
        if consecutive == SUSTAINED_OBSERVATIONS:
            crossing = point["target_interactions"]
            break
    return {"points": points, "normalized_success_auc": auc,
            "success_threshold": SUCCESS_THRESHOLD, "sustained_observations": SUSTAINED_OBSERVATIONS,
            "interactions_to_sustained_75_percent_success": crossing,
            "threshold_censored": crossing is None,
            "threshold_definition": "First second consecutive evaluation at >=75%; counted at confirmation, not backdated."}


def run_learning_to_learn(config, seed=7, output=None):
    output = Path(output or new_output("learning-to-learn"))
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    if not output.is_relative_to(ROOT / "runs"):
        raise ValueError("Transfer output must remain inside the new laboratory runs/")
    manifest = load_tasks(ROOT / config.get("task_manifest", "configs/tasks.json"))
    protocol = manifest["curricula"]["learning_to_learn"]
    target, target_steps, interval = protocol["target_task"], protocol["target_steps"], protocol["evaluation_interval"]
    if type(target_steps) is not int or target_steps <= 0 or type(interval) is not int or interval <= 0:
        raise ValueError("Target interactions and evaluation interval must be positive integers")
    expected_seeds = protocol["evaluation_seeds"]
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("Target evaluation seeds must be nonempty and unique")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "seed": seed, "target_task": target,
              "target_task_sha256": _task_hash(manifest["tasks"], target),
              "success_threshold": SUCCESS_THRESHOLD, "sustained_observations": SUSTAINED_OBSERVATIONS}
    write_json(output / "learning_to_learn.json", report)
    shared = {"task_id": target, "purpose": "target_curve", "evaluation_seeds": expected_seeds}
    target_stages = [{**shared, "train_steps": 0}] + [{**shared, "train_steps": min(interval, target_steps - i)}
                                                    for i in range(0, target_steps, interval)]
    results, tracker = {}, None
    try:
        for condition in ("fresh_same_initialization", "experienced"):
            current = copy.deepcopy(config)
            current["seed"] = seed
            current["architecture_id"] = config["architecture_id"] + "-" + condition
            sources = [copy.deepcopy(s) for s in protocol["stages"] if s.get("purpose") == "prior_experience"] if condition == "experienced" else []
            current["stages"] = sources + copy.deepcopy(target_stages)
            current["task_id"] = current["stages"][0]["task_id"]
            current["growth"] = {"kind": "none"}
            current.setdefault("resources", {})["max_seconds"] = 1200
            results[condition] = run_experiment(current, output / condition)
            if results[condition]["status"] != "FINISHED":
                raise RuntimeError(f"Transfer condition did not finish: {condition}")
            recorded_target = _task_hash(_saved_tasks(output / condition), target)
            if recorded_target != report["target_task_sha256"] or recorded_target is None:
                raise ValueError("Recorded target task content differs between transfer controls")
        hashes = [result["initial_trainable_sha256"] for result in results.values()]
        if None in hashes or len(set(hashes)) != 1:
            raise RuntimeError("Transfer controls did not start from the same initialization")
        frozen_hashes = [result.get("initial_frozen_sha256") for result in results.values()]
        if len(set(frozen_hashes)) != 1:
            raise RuntimeError("Transfer controls used different frozen parameters")
        curves, exposure, target_seed_streams = {}, {}, {}
        target_pool = manifest["tasks"][target]["seeds"]
        expected_training_seed_base = target_pool[0] + int(seed)
        expected_initial_training_seed = target_pool[int(seed) % len(target_pool)]
        for condition, result in results.items():
            stages = [stage for stage in result["stages"] if stage["purpose"] == "target_curve"]
            curves[condition] = _curve(stages, target, expected_seeds, target_steps)
            target_seed_streams[condition] = []
            for stage in stages:
                delta = stage["training_steps_after"] - stage["training_steps_before"]
                if not delta:
                    continue
                if stage.get("training_seed_base") != expected_training_seed_base:
                    raise ValueError("Target training seed bases differ between transfer controls")
                if stage.get("training_seed_pool") != manifest["tasks"][target]["seeds"]:
                    raise ValueError("Target training seed pools differ from the matched protocol")
                stream = stage.get("training_seeds_by_interaction")
                if not isinstance(stream, list) or len(stream) != delta or any(type(value) is not int or not 0 <= value < 2**32 for value in stream):
                    raise ValueError("Actual target training seed stream is missing or inconsistent")
                if not set(stream) <= set(stage["training_seed_pool"]):
                    raise ValueError("Actual target training used a seed outside its training pool")
                target_seed_streams[condition].extend(stream)
            if not target_seed_streams[condition] or target_seed_streams[condition][0] != expected_initial_training_seed:
                raise ValueError("Target initial training seed differs from the matched protocol")
            exposure[condition] = _exposure(result)
            if exposure[condition]["target_training_interactions"] != target_steps:
                raise ValueError("Actual target training exposure differs between controls")
        if exposure["fresh_same_initialization"]["source_training_interactions"] != 0:
            raise ValueError("Fresh control unexpectedly received source training")
        difference = curves["experienced"]["normalized_success_auc"] - curves["fresh_same_initialization"]["normalized_success_auc"]
        report.update(status="FINISHED", same_initialization=True, same_target_evaluation_seeds=True,
                      evaluation_seeds=expected_seeds, target_interactions_each=target_steps,
                      target_training_seed_base=expected_training_seed_base, same_target_training_seed_base=True,
                      target_training_seed_pool=target_pool, same_target_training_seed_pool=True,
                      target_initial_training_seed=expected_initial_training_seed,
                      target_training_seeds_by_interaction=target_seed_streams,
                      same_realized_target_seed_stream=len({_identity(stream) for stream in target_seed_streams.values()}) == 1,
                      target_training_seed_note="Same verified target seed base; adaptive actions/episode lengths can produce different realized map exposure. Actual seed per training interaction is recorded.",
                      curves=curves, exposure=exposure,
                      forward_transfer_auc_difference=difference, positive_transfer=max(0.0, difference), negative_transfer=min(0.0, difference),
                      claim="Single-seed infrastructure check. Does not establish meta-learning. Prior experience includes weights and optimizer state; its full interaction cost is reported.",
                      runs={condition: result["mlflow"] for condition, result in results.items()})
        tracker = _new_tracker(config.get("tracking", {}))
        report["mlflow"] = tracker.reference()
        tracker.params({"seed": seed, "target_task": target, "target_task_sha256": report["target_task_sha256"],
                        "target_interactions_each": target_steps, "same_initialization": True,
                        "success_threshold": SUCCESS_THRESHOLD, "sustained_observations": SUSTAINED_OBSERVATIONS})
        tracker.metrics({"forward_transfer_auc_difference": difference,
                         "fresh_auc": curves["fresh_same_initialization"]["normalized_success_auc"],
                         "experienced_auc": curves["experienced"]["normalized_success_auc"]})
        for condition, values in exposure.items():
            tracker.metrics(values, prefix=condition + ".")
        write_json(output / "learning_to_learn.json", report)
        tracker.client.log_artifact(tracker.run_id, str(output / "learning_to_learn.json"))
        tracker.client.set_terminated(tracker.run_id, status="FINISHED")
        return report
    except Exception as exc:
        report.update(status="FAILED", error=type(exc).__name__ + ": " + str(exc),
                      completed_conditions={condition: result.get("mlflow") for condition, result in results.items()})
        if tracker is not None:
            try:
                tracker.client.set_terminated(tracker.run_id, status="FAILED")
            except Exception as termination_error:
                report["tracking_termination_error"] = str(termination_error)
        write_json(output / "learning_to_learn.json", report)
        raise
