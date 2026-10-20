"""Boundary checks call production modules with expensive collaborators mocked."""

from __future__ import annotations

import fcntl
import json
import sys
import types
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "src/ucl"


@pytest.fixture
def guarded_runner(tmp_path, monkeypatch):
    from ucl import runner

    root = tmp_path / "new_lab"
    root.mkdir()

    class Guard:
        samples = []

        def __init__(self, *args):
            pass

        def check(self):
            pass

    def write_json(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "ResourceGuard", Guard)
    monkeypatch.setattr(runner, "write_json", write_json)
    monkeypatch.setattr(
        runner,
        "load_tasks",
        lambda _: {
            "tasks": {"heldout-navigation": {"id": "heldout-navigation", "split": "heldout"}}
        },
    )
    return runner.run_experiment, root


def test_runner_rejects_external_output_before_writing(guarded_runner, tmp_path):
    run, _ = guarded_runner
    outside = tmp_path / "outside"
    with pytest.raises(ValueError, match="inside"):
        run({"architecture_id": "test"}, outside)
    assert not outside.exists()


def test_runner_rejects_heldout_training_before_model_loading(guarded_runner):
    run, root = guarded_runner
    output = root / "runs/invalid-heldout"
    with pytest.raises(ValueError, match="development"):
        run({"architecture_id": "test", "task_id": "heldout-navigation"}, output)
    record = json.loads((output / "summary.json").read_text())
    assert record["status"] == "FAILED"
    assert "development" in record["error"]
    assert not (output / "checkpoints").exists()


def test_runner_refuses_concurrent_execution(guarded_runner):
    run, root = guarded_runner
    (root / "runs").mkdir()
    with (root / "runs/.execution.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="sequential"):
            run({"architecture_id": "test"}, root / "runs/blocked")
    assert not (root / "runs/blocked/checkpoints").exists()


def test_runner_rejects_unknown_mode_before_writing(guarded_runner):
    run, root = guarded_runner
    output = root / "runs/invalid-mode"
    with pytest.raises(ValueError, match="mode"):
        run({"architecture_id": "test"}, output, mode="pretend")
    assert not output.exists()


def test_cli_routes_evaluation_overrides_without_loading_models(tmp_path, monkeypatch):
    from ucl import cli

    calls = []
    fake = types.ModuleType("ucl.runner")
    fake.run_experiment = lambda config, output, **kwargs: (
        calls.append((config, output, kwargs)) or {}
    )
    monkeypatch.setitem(sys.modules, "ucl.runner", fake)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"seed": 1, "train_steps": 128}))
    assert (
        cli.main(
            [
                "evaluate",
                "--config",
                str(config),
                "--task",
                "development-navigation",
                "--seed",
                "19",
                "--checkpoint",
                "runs/test/last.pt",
            ]
        )
        == 0
    )
    assert calls[0][0]["seed"] == 19
    assert calls[0][2] == {
        "mode": "evaluate",
        "checkpoint": "runs/test/last.pt",
        "task_id": "development-navigation",
        "curriculum": None,
    }
    assert json.loads(config.read_text())["seed"] == 1


def test_cli_negative_steps_never_calls_runner(tmp_path, monkeypatch):
    from ucl import cli

    fake = types.ModuleType("ucl.runner")
    fake.run_experiment = lambda *args, **kwargs: pytest.fail("Invalid request reached runner")
    monkeypatch.setitem(sys.modules, "ucl.runner", fake)
    config = tmp_path / "config.json"
    config.write_text("{}")
    with pytest.raises(SystemExit) as error:
        cli.main(["train", "--config", str(config), "--steps", "-1"])
    assert error.value.code == 2


def test_cli_resume_requires_checkpoint():
    from ucl import cli

    with pytest.raises(SystemExit) as error:
        cli.main(["resume"])
    assert error.value.code == 2


def test_transfer_report_rejects_external_output_before_writing(tmp_path, monkeypatch):
    from ucl import reports

    root = tmp_path / "new_lab"
    root.mkdir()
    monkeypatch.setattr(reports, "ROOT", root)
    outside = tmp_path / "outside"
    with pytest.raises(ValueError, match="inside"):
        reports.run_learning_to_learn({"architecture_id": "test"}, output=outside)
    assert not outside.exists()


def test_checkpoint_selection_rejects_different_component():
    from ucl.checkpoints import validate_selected_architecture

    lstm = {
        "kind": "graph",
        "nodes": [{"id": "memory", "type": "lstm", "params": {"output_dim": 32}}],
    }
    gru = {
        "kind": "graph",
        "nodes": [{"id": "memory", "type": "gru", "params": {"output_dim": 32}}],
    }
    with pytest.raises(ValueError, match="differs"):
        validate_selected_architecture(lstm, gru)


def test_checkpoint_selection_accepts_only_bounded_growth():
    import copy

    from ucl.checkpoints import validate_selected_architecture

    selected = {
        "kind": "graph",
        "nodes": [
            {
                "id": "memory",
                "type": "expandable_rim",
                "params": {"modules": 2, "max_modules": 4, "output_dim": 32},
            }
        ],
    }
    restored = copy.deepcopy(selected)
    restored["nodes"][0]["params"]["modules"] = 3
    validate_selected_architecture(selected, restored)
    assert selected["nodes"][0]["params"]["modules"] == 2
    restored["nodes"][0]["params"]["modules"] = 5
    with pytest.raises(ValueError, match="limits"):
        validate_selected_architecture(selected, restored)


def comparison_fixture(root, name, seed=7, task="heldout-navigation", steps=32):
    directory = root / "runs" / name
    directory.mkdir(parents=True)
    row = {
        "status": "FINISHED",
        "architecture_id": "EXP-002",
        "config": {"seed": seed},
        "training_steps": steps,
        "metrics": {
            "task_id": task,
            "seeds": [240000, 240001],
            "success_rate": 0.5,
            "reward_mean": 0.2,
        },
        "elapsed_seconds": 1,
        "initial_parameters": {"trainable": 100, "frozen": 0},
        "final_parameters": {"trainable": 100, "frozen": 0},
        "growth_events": [],
    }
    (directory / "summary.json").write_text(json.dumps(row))
    return directory


def test_comparison_does_not_pool_different_tasks(tmp_path, monkeypatch):
    from ucl import reports

    monkeypatch.setattr(reports, "ROOT", tmp_path)
    a = comparison_fixture(tmp_path, "a")
    b = comparison_fixture(tmp_path, "b", task="heldout-pickup")
    result = reports.comparison_report([a, b], tmp_path / "runs/comparison.json")
    assert result["same_evaluation_task_and_seeds"] is False
    assert len(result["aggregates"]) == 2
    assert all(row["std_between_seeds"] is None for row in result["aggregates"])


def test_comparison_rejects_duplicate_seed_as_independent_sample(tmp_path, monkeypatch):
    from ucl import reports

    monkeypatch.setattr(reports, "ROOT", tmp_path)
    a = comparison_fixture(tmp_path, "a")
    b = comparison_fixture(tmp_path, "b")
    with pytest.raises(ValueError, match="Repeated seed"):
        reports.comparison_report([a, b], tmp_path / "runs/comparison.json")
    assert not (tmp_path / "runs/comparison.json").exists()
