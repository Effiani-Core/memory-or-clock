"""STUDY-001 runner on tiny hand-built 7x7 panels; never reads the real study panels."""
from __future__ import annotations

import copy
import json
import sys
import types

import pytest
import torch

from ucl import study as study_module
from ucl.study import acquisition_step, normalized_auc, retention_summary, run_study
from ucl.tasks import load_tasks, ruleset_identity, task_identity

SOURCES = {"A": "development-navigation", "B": "development-pickup",
           "C": "development-hold_transform"}
PROTOCOL = {"study_id": "UCL-STUDY-001-TEST",
            "algorithm": {"name": "a2c", "lr": 0.001, "rollout_steps": 16, "gamma": 0.99,
                          "entropy_coef": 0.01},
            "single_map": {"steps": 64, "eval_every": 32, "episodes": 4, "threshold": 0.9},
            "pilot": {"steps_per_task": 64, "eval_every": 32, "initializations": [7]},
            "acquisition": {"threshold": 0.75, "consecutive": 2, "min_gain": 0.2},
            "resources": {"max_seconds": 600, "min_ram_mib": 100, "max_temperature_c": 95,
                          "min_disk_gib": 0},
            "resource_check_every_updates": 2,
            "tracking": {"uri": "http://127.0.0.1:9", "experiment": "Universal Cognitive Lab"}}
ARCHITECTURE = {"architecture_id": "TEST-MLP", "name": "mlp_no_memory", "architecture": {
    "kind": "graph", "output": "encoder", "max_parameters": 250000,
    "nodes": [{"id": "encoder", "type": "mlp", "inputs": ["observation"],
               "params": {"output_dim": 8}}]}}


def point(step, success):
    return {"step": step, "success_rate": success}


def test_acquisition_needs_consecutive_points_and_records_the_completing_one():
    curve = [point(0, 0.0), point(10, 0.8), point(20, 0.5), point(30, 0.8), point(40, 0.9)]
    assert acquisition_step(curve, 0.0, 0.0, 0.75, 2, 0.2) == 40
    assert acquisition_step(curve, 0.0, 0.0, 0.75, 1, 0.2) == 10
    assert acquisition_step(curve[:4], 0.0, 0.0, 0.75, 2, 0.2) is None


def test_acquisition_requires_gain_over_initial_and_random():
    curve = [point(0, 0.7), point(10, 0.8), point(20, 0.85)]
    assert acquisition_step(curve, 0.7, 0.0, 0.75, 2, 0.2) is None
    assert acquisition_step(curve, 0.0, 0.7, 0.75, 2, 0.2) is None
    exact = [point(0, 0.55), point(10, 0.75), point(20, 0.75)]
    assert acquisition_step(exact, 0.55, 0.0, 0.75, 2, 0.2) == 20


def test_normalized_auc_uses_trapezoids_over_the_step_span():
    assert normalized_auc([]) == 0.0 and normalized_auc([point(0, 1.0)]) == 0.0
    assert normalized_auc([point(0, 0.0), point(10, 1.0)]) == pytest.approx(0.5)
    assert normalized_auc([point(0, 0.0), point(10, 1.0), point(30, 1.0)]) == pytest.approx(
        (5 + 20) / 30)


def test_retention_is_measured_from_end_of_stage_rows():
    def row(**success):
        return {"test": {key: {"success_rate": value} for key, value in success.items()}}
    matrix = {"after_A": row(A=0.8, B=0.1), "after_B": row(A=0.5, B=0.9)}
    result = retention_summary(matrix, ["A", "B"], {"A": {"step": 32}})
    assert result["S_A_A"] == 0.8 and result["S_A_last"] == 0.5
    assert result["change_pp"] == pytest.approx(-30) and result["forgetting"] == pytest.approx(0.3)
    assert result["interpretable"] and result["backward_transfer_pp"]["A"] == pytest.approx(-30)
    assert retention_summary(matrix, ["A"], {"A": {"step": None}}) == {}


def tiny_study():
    source = load_tasks()["tasks"]
    tasks, task_ids, panels = {}, {}, {}
    for index, (key, name) in enumerate(SOURCES.items()):
        task = copy.deepcopy(source[name])
        task.update(id=f"study-test-{task['family']}", height=7, width=7)
        task["task_sha256"], task["ruleset_sha256"] = task_identity(task), ruleset_identity(task)
        base = 9_000_000 + 100_000 * index
        tasks[task["id"]], task_ids[key] = task, task["id"]
        panels[task["id"]] = {"train": [base, base + 1, base + 2],
                              "validation": [base + 10, base + 11],
                              "test": [base + 20, base + 21], "final": [base + 30, base + 31],
                              "known": [base]}
    return {"protocol": copy.deepcopy(PROTOCOL), "tasks": tasks, "panels": panels,
            "task_ids": task_ids, "protocol_hash": "test-hash"}


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    return tmp_path_factory.mktemp("lab"), tiny_study()


@pytest.fixture
def seen(monkeypatch):
    seeds = []
    original = study_module.evaluate_episode

    def recording(model, task, seed, **kwargs):
        seeds.append(int(seed))
        return original(model, task, seed, **kwargs)
    monkeypatch.setattr(study_module, "evaluate_episode", recording)
    return seeds


def final_seeds(study):
    return {seed for panel in study["panels"].values() for seed in panel["final"]}


def test_two_stage_run_writes_checkpoints_curves_matrix_and_retention(lab, seen):
    root, study = lab
    output = root / "runs/study_001/ab"
    summary = run_study(study, ARCHITECTURE, 7, ["A", "B"], steps=64, eval_every=32,
                        output=output, root=root)
    assert summary["status"] == "FINISHED" and summary["phase"] == "sequence"
    for name in ("initial", "best_A", "after_A", "best_B", "after_B"):
        assert (output / f"checkpoints/{name}.pt").is_file()
        assert len(summary["checkpoints"][name]["sha256"]) == 64
    assert not (output / "checkpoints/interrupted.pt").exists()
    written = json.loads((output / "summary.json").read_text())
    assert written["status"] == "FINISHED" and written["protocol_hash"] == "test-hash"
    curve = [json.loads(line) for line in (output / "curve.jsonl").read_text().splitlines()]
    assert [(p["task_key"], p["step"]) for p in curve] == [
        ("A", 0), ("A", 32), ("A", 64), ("B", 0), ("B", 32), ("B", 64)]
    rows = [json.loads(line) for line in (output / "train_metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 8 and "training_seeds" not in rows[0] and "loss" in rows[0]
    assert set(written["matrix"]) == {"initial", "after_A", "after_B"}
    assert set(written["matrix"]["after_A"]["test"]) == {"A", "B", "C"}
    assert written["matrix"]["after_B"]["sha256"] == written["checkpoints"]["after_B"]["sha256"]
    assert set(written["random_baselines"]["C"]) == {"validation", "test"}
    retention = written["retention"]
    assert retention["S_A_A"] == written["matrix"]["after_A"]["test"]["A"]["success_rate"]
    assert retention["S_A_last"] == written["matrix"]["after_B"]["test"]["A"]["success_rate"]
    assert {"change_pp", "forgetting", "forgetting_pp", "interpretable"} <= set(retention)
    assert set(written["acquisition"]) == set(written["auc"]) == {"A", "B"}
    assert written["interactions"]["training"] == 128 and written["interactions"]["evaluation"] > 0
    assert written["mlflow"]["isolated_fallback"] is True
    assert written["source_identity"] and "final_panel" not in written
    assert not final_seeds(study) & set(seen)
    after_a = torch.load(output / "checkpoints/after_A.pt", weights_only=True)
    assert after_a["metadata"]["role"] == "end_of_stage" and after_a["trainer"]["total_steps"] == 64


def test_every_evaluation_keeps_its_own_checkpoint(lab, seen):
    root, study = lab
    output = root / "runs/study_001/eval-checkpoints"
    summary = run_study(study, ARCHITECTURE, 7, ["A", "B"], steps=64, eval_every=32,
                        output=output, root=root)
    curve = [json.loads(line) for line in (output / "curve.jsonl").read_text().splitlines()]
    saved = summary["evaluation_checkpoints"]
    assert [p["checkpoint"] for p in curve] == [
        f"checkpoints/evaluations/{key}_{step:07d}.pt" for key in "AB" for step in (0, 32, 64)]
    assert [entry["path"] for entry in saved.values()] == [p["checkpoint"] for p in curve]
    assert set(summary["checkpoints"]) == {"initial", "best_A", "after_A", "best_B", "after_B"}
    last = torch.load(output / saved["B_0000064"]["path"], weights_only=True)
    after_b = torch.load(output / "checkpoints/after_B.pt", weights_only=True)
    assert last["metadata"]["role"] == "evaluation" and last["metadata"]["stage_step"] == 64
    assert last["metadata"]["validation_success"] == curve[-1]["success_rate"]
    assert all(torch.equal(last["weights"][name], after_b["weights"][name])
               for name in after_b["weights"])


def test_full_sequence_evaluates_final_panel_only_on_request(lab, seen):
    root, study = lab
    summary = run_study(study, ARCHITECTURE, 19, ["A", "B", "C"], steps=16, eval_every=16,
                        output=root / "runs/study_001/abc-final", root=root,
                        final_evaluation=True)
    assert summary["status"] == "FINISHED"
    assert set(summary["final_panel"]["results"]) == {"A", "B", "C"}
    assert summary["final_panel"]["checkpoint"] == summary["checkpoints"]["after_C"]
    assert final_seeds(study) <= set(seen)
    assert summary["retention"]["after_stage"] == "C"


def test_single_map_run_trains_and_evaluates_one_map(lab, seen):
    root, study = lab
    output = root / "runs/study_001/single"
    summary = run_study(study, ARCHITECTURE, 7, ["A"], steps=64, eval_every=32, output=output,
                        root=root, single_map=True)
    one_map = study["panels"][study["task_ids"]["A"]]["train"][0]
    assert summary["phase"] == "single_map" and summary["matrix"] == {}
    assert set(seen) == {one_map}
    assert list(summary["random_baselines"]) == ["A"]
    assert list(summary["random_baselines"]["A"]) == ["single_map"]
    assert summary["curves"]["A"][0]["episodes"] == PROTOCOL["single_map"]["episodes"]
    assert {"passed", "final_success", "threshold"} <= set(summary["single_map_check"])
    assert summary["stages"][0]["seed_pool_size"] == 1
    assert (output / "checkpoints/after_A.pt").is_file() and "retention" not in summary


def test_failure_is_recorded_with_interrupted_checkpoint_and_reraised(lab, monkeypatch):
    root, study = lab
    output = root / "runs/study_001/failed"

    def broken(self, steps=None):
        raise RuntimeError("synthetic training failure")
    monkeypatch.setattr(study_module.A2CTrainer, "update", broken)
    with pytest.raises(RuntimeError, match="synthetic"):
        run_study(study, ARCHITECTURE, 7, ["A"], steps=16, eval_every=16, output=output,
                  root=root)
    written = json.loads((output / "summary.json").read_text())
    assert written["status"] == "FAILED" and "synthetic" in written["error"]
    assert (output / "checkpoints/interrupted.pt").is_file()


@pytest.mark.parametrize("keys,kwargs,match", [
    (["A", "B"], {"final_evaluation": True}, "final panel"),
    (["A", "B"], {"single_map": True}, "exactly one"),
    (["A", "A"], {}, "distinct"),
    (["D"], {}, "distinct"),
])
def test_invalid_requests_fail_before_writing(tmp_path, keys, kwargs, match):
    with pytest.raises(ValueError, match=match):
        run_study(tiny_study(), ARCHITECTURE, 7, keys, steps=16, eval_every=16,
                  output=tmp_path / "runs/study_001/x", root=tmp_path, **kwargs)
    assert not (tmp_path / "runs").exists()


def test_output_must_be_new_and_inside_study_runs(tmp_path):
    with pytest.raises(ValueError, match="inside runs/study_001"):
        run_study(tiny_study(), ARCHITECTURE, 7, ["A"], steps=16, eval_every=16,
                  output=tmp_path / "runs/elsewhere", root=tmp_path)
    existing = tmp_path / "runs/study_001/existing"
    existing.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        run_study(tiny_study(), ARCHITECTURE, 7, ["A"], steps=16, eval_every=16,
                  output=existing, root=tmp_path)


def test_study_run_cli_uses_protocol_defaults(monkeypatch, capsys):
    from ucl import cli
    calls = {}
    fake = types.ModuleType("ucl.study_panels")
    fake.load_study = lambda directory: {"protocol": PROTOCOL}
    monkeypatch.setitem(sys.modules, "ucl.study_panels", fake)

    def fake_run(study, architecture, seed, keys, **kwargs):
        calls.update(architecture=architecture["architecture_id"], seed=seed, keys=keys, **kwargs)
        return {"status": "FINISHED", "output": "x"}
    monkeypatch.setattr(study_module, "run_study", fake_run)
    assert cli.main(["study-run", "--arch", "configs/study_001/arch-MLP.json", "--seed", "19",
                     "--tasks", "A,B"]) == 0
    assert calls["keys"] == ["A", "B"] and calls["seed"] == 19
    assert (calls["steps"], calls["eval_every"]) == (64, 32) and not calls["single_map"]
    assert cli.main(["study-run", "--single-map", "--steps", "48"]) == 0
    assert calls["single_map"] and calls["steps"] == 48 and calls["architecture"] == "S001-LSTM"
    assert calls["eval_every"] == PROTOCOL["single_map"]["eval_every"]
    assert '"status": "FINISHED"' in capsys.readouterr().out


def test_study_pilot_runs_each_combination_sequentially(monkeypatch):
    from ucl import cli
    commands = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kwargs: commands.append(
        (command, kwargs["check"])))
    assert cli.main(["study-pilot", "--archs", "LSTM,MLP", "--tasks", "A,B", "--seeds",
                     "7,19"]) == 0
    assert len(commands) == 8 and all(check for _, check in commands)
    first = commands[0][0]
    assert first[first.index("study-run") + 1:] == [
        "--arch", "configs/study_001/arch-LSTM.json", "--seed", "7", "--tasks", "A"]
    with pytest.raises(SystemExit):
        cli.main(["study-pilot", "--archs", "NOPE"])


def test_an_architecture_can_override_only_the_learning_rate(lab):
    root, study = lab
    tuned = {**ARCHITECTURE, "algorithm": {"lr": 0.003}}
    summary = run_study(study, tuned, 7, ["A"], steps=16, eval_every=16,
                        output=root / "runs/study_001/lr", root=root)
    assert summary["algorithm"]["lr"] == 0.003 and summary["algorithm"]["gamma"] == 0.99
    with pytest.raises(ValueError, match="only override"):
        run_study(study, {**ARCHITECTURE, "algorithm": {"gamma": 0.5}}, 7, ["A"], steps=16,
                  eval_every=16, output=root / "runs/study_001/bad", root=root)
