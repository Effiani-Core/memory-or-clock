"""Synthetic report fixtures test accounting, not learning performance."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from ucl import reports
from ucl.tasks import task_identity


def task(identifier, color=1, seed=100):
    result = {"id": identifier, "adapter": "xland", "env_id": "XLand-MiniGrid-R1-9x9",
              "height": 5, "width": 5, "view_size": 5, "max_steps": 32,
              "goal": {"type": "AgentOnTileGoal", "tile": [6, color]},
              "rules": [], "init_tiles": [[6, color]], "seeds": list(range(seed, seed + 16)),
              "evaluation_seeds": [900, 901], "split": "development", "family": identifier}
    result["task_sha256"] = task_identity(result)
    return result


def curve_stage(position, score, previous=None, seeds=None):
    return {"purpose": "target_curve", "task_id": "target", "seeds": seeds or [900, 901],
            "episodes": 2, "environment_steps": 6, "training_steps_before": position if previous is None else previous,
            "training_steps_after": position, "success_rate": score, "reward_mean": score / 2}


def test_threshold_is_75_percent_confirmed_at_second_observation():
    stages = [curve_stage(0, 0.75), curve_stage(4, 0.75, 0), curve_stage(8, 1.0, 4)]
    result = reports._curve(stages, "target", [900, 901], 8)
    assert result["interactions_to_sustained_75_percent_success"] == 4
    assert not result["threshold_censored"]
    assert result["success_threshold"] == 0.75
    assert result["sustained_observations"] == 2
    assert "interactions_to_50_percent_success" not in result


@pytest.mark.parametrize("scores", [[0.5, 0.5, 0.5], [0.75, 0.5, 0.75], [0.0, 0.5, 1.0]])
def test_one_crossing_or_50_percent_is_censored(scores):
    result = reports._curve([curve_stage(i * 4, score) for i, score in enumerate(scores)], "target", [900, 901], 8)
    assert result["interactions_to_sustained_75_percent_success"] is None
    assert result["threshold_censored"]


def test_normalized_auc_uses_actual_target_interactions():
    result = reports._curve([curve_stage(0, 0), curve_stage(3, 0.5), curve_stage(8, 1)], "target", [900, 901], 8)
    assert result["normalized_success_auc"] == (3 * 0.25 + 5 * 0.75) / 8


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows[1].update(seeds=[900, 902]), "evaluation seeds"),
    (lambda rows: rows[1].update(task_id="another"), "Target task"),
    (lambda rows: rows[1].update(episodes=1), "episode count"),
    (lambda rows: rows[0].update(training_steps_after=1), "start at zero"),
    (lambda rows: rows[-1].update(training_steps_after=9), "requested target exposure"),
])
def test_target_curve_rejects_unmatched_protocol(mutation, match):
    stages = [curve_stage(0, 0), curve_stage(4, 0.5), curve_stage(8, 0.75)]
    mutation(stages)
    with pytest.raises(ValueError, match=match):
        reports._curve(stages, "target", [900, 901], 8)


def test_exposure_counts_source_target_and_all_actual_evaluations():
    source = {"purpose": "prior_experience", "training_steps_before": 0, "training_steps_after": 8,
              "environment_steps": 4, "episodes": 1}
    result = {"mode": "train", "training_steps": 16, "new_training_steps": 16,
              "stages": [source, curve_stage(8, 0), curve_stage(12, 0.5, 8), curve_stage(16, 1, 12)],
              "evaluations": [{"environment_steps": 7, "episodes": 2}],
              "metrics": {"environment_steps": 5, "episodes": 2}}
    exposure = reports._exposure(result)
    assert exposure["source_training_interactions"] == 8
    assert exposure["target_training_interactions"] == 8
    assert exposure["total_evaluation_interactions"] == 4 + 18 + 7 + 5
    assert exposure["total_environment_interactions"] == 16 + 34
    assert exposure["evaluation_episodes"] == 11
    result["mode"] = "evaluate"
    assert reports._exposure(result)["total_evaluation_interactions"] == 29
    result["new_training_steps"] = 17
    with pytest.raises(ValueError, match="Stage exposure"):
        reports._exposure(result)


def comparison_run(root, name, *, seed=7, lr=0.001, color=1, source="revision-a"):
    directory = root / "runs" / name
    directory.mkdir(parents=True)
    manifest = {"tasks": {"target": task("target", color)}}
    configuration = {"seed": seed, "architecture_id": "EXP-002", "task_id": "target",
                     "evaluation_task": "target", "heldout_task": "target",
                     "architecture": {"kind": "graph", "nodes": [{"type": "lstm"}]},
                     "algorithm": {"name": "a2c", "lr": lr}, "source_identity": source}
    summary = {"status": "FINISHED", "architecture_id": "EXP-002", "config": configuration,
               "training_steps": 8, "new_training_steps": 8, "source_identity": source,
               "metrics": {"task_id": "target", "seeds": [900, 901], "success_rate": 0.5, "reward_mean": 0.1},
               "elapsed_seconds": 2, "initial_parameters": {"trainable": 100, "frozen": 0},
               "final_parameters": {"trainable": 100, "frozen": 0}, "growth_events": []}
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "tasks_manifest.json").write_text(json.dumps(manifest))
    return directory


def test_cohort_pools_replicate_seeds_not_source_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "ROOT", tmp_path)
    a = comparison_run(tmp_path, "a", seed=7, source="a")
    b = comparison_run(tmp_path, "b", seed=19, source="b")
    result = reports.comparison_report([a, b], tmp_path / "runs/report.json")
    assert len(result["aggregates"]) == 1
    assert result["aggregates"][0]["seeds"] == [7, 19]
    assert result["aggregates"][0]["source_identities"] == ["a", "b"]
    assert result["aggregates"][0]["std_between_seeds"] == 0
    assert result["same_evaluation_task_and_seeds"]


@pytest.mark.parametrize("difference", [{"lr": 0.1}, {"color": 2}])
def test_cohort_separates_changed_algorithm_or_same_label_different_task(tmp_path, monkeypatch, difference):
    monkeypatch.setattr(reports, "ROOT", tmp_path)
    a = comparison_run(tmp_path, "a", seed=7)
    b = comparison_run(tmp_path, "b", seed=19, **difference)
    result = reports.comparison_report([a, b], tmp_path / "runs/report.json")
    assert len(result["aggregates"]) == 2
    if "color" in difference:
        assert not result["same_evaluation_task_and_seeds"]


def test_cohort_rejects_duplicate_seed_and_corrupted_task_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "ROOT", tmp_path)
    a = comparison_run(tmp_path, "a")
    b = comparison_run(tmp_path, "b", source="other-source")
    with pytest.raises(ValueError, match="Repeated seed"):
        reports.comparison_report([a, b], tmp_path / "runs/report.json")
    manifest = json.loads((a / "tasks_manifest.json").read_text())
    manifest["tasks"]["target"]["goal"]["tile"][1] = 2
    (a / "tasks_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity mismatch"):
        reports.comparison_report([a], tmp_path / "runs/report.json")


class FakeTracker:
    def __init__(self, failure=None):
        self.client = self
        self.run_id = "synthetic-tracker"
        self.failure = failure
        self.terminations = []
    def params(self, values):
        if self.failure == "params":
            raise RuntimeError("synthetic params failure")
    def metrics(self, values, **kwargs):
        pass
    def reference(self):
        return {"run_id": self.run_id}
    def log_artifact(self, *args):
        if self.failure == "artifact":
            raise RuntimeError("synthetic artifact failure")
    def set_terminated(self, run_id, status):
        if self.failure == "termination" and status == "FINISHED":
            raise RuntimeError("synthetic termination failure")
        self.terminations.append(status)


@pytest.fixture
def transfer_harness(tmp_path, monkeypatch):
    manifest = {"tasks": {"source": task("source", 2, 100), "target": task("target", 1, 200)},
                "curricula": {"learning_to_learn": {"target_task": "target", "target_steps": 8,
                    "evaluation_interval": 4, "evaluation_seeds": [900, 901],
                    "stages": [{"task_id": "source", "purpose": "prior_experience", "train_steps": 8}]}}}
    monkeypatch.setattr(reports, "ROOT", tmp_path)
    monkeypatch.setattr(reports, "load_tasks", lambda path: deepcopy(manifest))
    tracker = FakeTracker()
    monkeypatch.setattr(reports, "_new_tracker", lambda config: tracker)
    mutate = {}
    def run(config, directory):
        experienced = config["architecture_id"].endswith("-experienced")
        directory.mkdir()
        (directory / "tasks_manifest.json").write_text(json.dumps(manifest))
        stages, total, target_started = [], 0, False
        for stage in config["stages"]:
            delta = stage["train_steps"]
            point = {"task_id": stage["task_id"], "purpose": stage["purpose"], "training_steps_before": total,
                     "training_steps_after": total + delta, "episodes": 2, "environment_steps": 6,
                     "seeds": [900, 901], "success_rate": 0.75, "reward_mean": 0.25,
                     "training_seed_base": 207 if delta and stage["purpose"] == "target_curve" else None,
                     "training_seed_pool": manifest["tasks"]["target"]["seeds"] if delta and stage["purpose"] == "target_curve" else None,
                     "training_seeds_by_interaction": ([208] if experienced else [207]) * delta}
            if delta and stage["purpose"] == "target_curve" and not target_started:
                point["training_seeds_by_interaction"][0] = 207
                target_started = True
            total += delta
            stages.append(point)
        result = {"status": "FINISHED", "initial_trainable_sha256": "same", "initial_frozen_sha256": "same-frozen",
                  "training_steps": total, "new_training_steps": total, "stages": stages,
                  "evaluations": [{"environment_steps": 3, "episodes": 2}],
                  "metrics": {"environment_steps": 5, "episodes": 2}, "mlflow": {"run_id": config["architecture_id"]}}
        if experienced and "callback" in mutate:
            mutate["callback"](result)
        return result
    monkeypatch.setattr(reports, "run_experiment", run)
    return tmp_path, tracker, mutate


def test_learning_to_learn_reports_actual_cost_and_seed_streams(transfer_harness):
    root, tracker, _ = transfer_harness
    result = reports.run_learning_to_learn({"architecture_id": "EXP-002"}, output=root / "runs/transfer")
    assert result["status"] == "FINISHED"
    assert result["target_interactions_each"] == 8
    assert result["same_target_training_seed_base"]
    assert not result["same_realized_target_seed_stream"]
    assert result["exposure"]["fresh_same_initialization"]["total_environment_interactions"] == 8 + 18 + 3 + 5
    assert result["exposure"]["experienced"]["total_environment_interactions"] == 16 + 24 + 3 + 5
    assert result["curves"]["experienced"]["interactions_to_sustained_75_percent_success"] == 4
    assert tracker.terminations == ["FINISHED"]


@pytest.mark.parametrize("failure", ["params", "artifact", "termination"])
def test_transfer_tracking_failure_marks_run_and_local_report_failed(transfer_harness, failure):
    root, tracker, _ = transfer_harness
    tracker.failure = failure
    output = root / "runs/transfer"
    with pytest.raises(RuntimeError, match="synthetic"):
        reports.run_learning_to_learn({"architecture_id": "EXP-002"}, output=output)
    assert tracker.terminations == ["FAILED"]
    result = json.loads((output / "learning_to_learn.json").read_text())
    assert result["status"] == "FAILED"
    assert len(result["completed_conditions"]) == 2


@pytest.mark.parametrize("change,match", [
    (lambda result: result["stages"][-1].update(seeds=[901, 902]), "evaluation seeds"),
    (lambda result: result["stages"][-1].update(training_seed_base=999), "training seed bases"),
    (lambda result: result["stages"][-1].update(training_seeds_by_interaction=[]), "seed stream"),
    (lambda result: result["stages"][-1].update(training_seed_pool=[999]), "seed pools"),
    (lambda result: result["stages"][-1].update(training_seeds_by_interaction=[900] * 4), "outside its training pool"),
])
def test_transfer_rejects_unmatched_target_seeds(transfer_harness, change, match):
    root, tracker, mutate = transfer_harness
    mutate["callback"] = change
    with pytest.raises(ValueError, match=match):
        reports.run_learning_to_learn({"architecture_id": "EXP-002"}, output=root / "runs/transfer")
    assert not tracker.terminations
    assert json.loads((root / "runs/transfer/learning_to_learn.json").read_text())["status"] == "FAILED"
