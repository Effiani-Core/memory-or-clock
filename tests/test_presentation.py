"""Saved-model menus use metadata only; no neural framework or model loading."""

import json
import os
import subprocess
import sys
from copy import deepcopy

from ucl.presentation import (
    architecture_description,
    architecture_fingerprint,
    architecture_name,
    compatible_checkpoints,
    run_catalog,
    split_name,
    task_description,
    task_name,
)


def architecture(kind="lstm", width=8):
    return {
        "kind": "graph",
        "nodes": [
            {
                "id": "memory",
                "type": kind,
                "inputs": ["observation"],
                "params": {"output_dim": width},
            }
        ],
        "output": "memory",
    }


def make_run(
    root, name, identifier="EXP-002", selected=None, mtime=100, status="FINISHED", mirror=False
):
    directory = root / "runs" / ("mlartifacts" if mirror else "") / name
    directory.mkdir(parents=True)
    config = {
        "architecture_id": identifier,
        "architecture": selected or architecture(),
        "seed": 7,
        "task_id": "development-navigation",
    }
    summary = {
        "status": status,
        "architecture_id": identifier,
        "config": config,
        "mode": "train",
        "training_steps": 32,
        "metrics": {"task_id": "heldout-navigation"},
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "checkpoints").mkdir()
    for kind in ("last", "best", "initial"):
        (directory / "checkpoints" / f"{kind}.pt").write_bytes(
            b"not-torch: menu must not load tensors"
        )
    os.utime(directory / "summary.json", (mtime, mtime))
    return directory, config


def test_human_names_explain_without_promising_learning():
    assert architecture_name("EXP-002") == "Memoria LSTM"
    assert "azar" in architecture_name("EXP-001")
    assert "experiencia previa" in architecture_name("EXP-002-experienced")
    assert "no aprende" in architecture_description("EXP-001")
    assert architecture_name("new_architecture") == "New architecture"
    assert split_name("heldout") == "Prueba final"
    assert task_name("heldout-navigation") == "Llegar al objetivo · Prueba final"
    assert "transform" in task_description("development-composition").lower()


def test_exact_goal_description_is_observer_only_and_uses_real_colors():
    task = {"id": "heldout-navigation", "goal": {"type": "AgentOnTileGoal", "tile": [6, 10]}}
    before = deepcopy(task)
    assert task_description(task) == "Llega a la casilla de color café."
    task = {"family": "hold_transform", "goal": {"type": "AgentHoldGoal", "tile": [11, 2]}}
    assert task_description(task).startswith("Recoge y conserva el hexágono de color verde.")
    assert "transformarlo" in task_description(task)
    assert before["goal"]["tile"] == [6, 10]


def test_catalog_excludes_mirrors_and_audited_provisional_run(tmp_path):
    make_run(tmp_path, "original", mtime=100)
    make_run(tmp_path, "mirrored", mirror=True, mtime=200)
    make_run(tmp_path, "progressive-smoke", mtime=300)
    (tmp_path / "audit").mkdir()
    (tmp_path / "audit/execution_provenance.json").write_text(
        json.dumps({"provisional_run": "runs/progressive-smoke"})
    )
    rows, errors = run_catalog(tmp_path)
    assert not errors
    assert [row["run_id"] for row in rows] == ["original"]
    assert rows[0]["path"].name == "summary.json"
    assert "Memoria LSTM" in rows[0]["label"]


def test_checkpoint_filter_rejects_other_architecture_and_changed_width(tmp_path):
    _, config = make_run(tmp_path, "correct", mtime=100)
    make_run(tmp_path, "other_id", identifier="EXP-003", selected=architecture("gru"), mtime=400)
    make_run(tmp_path, "wrong_width", selected=architecture(width=16), mtime=300)
    make_run(tmp_path, "same_arch_other_control", identifier="CONTROL-RIM-MAX", mtime=200)
    choices, errors = compatible_checkpoints(tmp_path, config)
    assert not errors
    assert len(choices) == 3
    assert {choice["run_id"] for choice in choices} == {"correct"}
    assert [choice["kind"] for choice in choices] == ["last", "best", "initial"]
    assert choices[0]["steps"] == 32
    assert choices[1]["steps"] is None
    assert "no garantiza aprendizaje" in choices[1]["description"]
    assert "/" not in choices[0]["label"].split(" · ")[0]


def test_default_is_latest_last_and_does_not_load_checkpoint_bytes(tmp_path):
    _, config = make_run(tmp_path, "older", mtime=100)
    make_run(tmp_path, "newer", mtime=200)
    choices, errors = compatible_checkpoints(tmp_path, config)
    assert not errors
    assert choices[0]["run_id"] == "newer" and choices[0]["kind"] == "last"
    assert choices[0]["path"].read_bytes().startswith(b"not-torch")
    assert all(choice["summary"]["status"] == "FINISHED" for choice in choices)


def test_bounded_expansion_accepted_but_other_settings_rejected(tmp_path):
    selected = architecture("expandable_rim")
    selected["nodes"][0]["params"].update(modules=2, max_modules=4)
    grown = deepcopy(selected)
    grown["nodes"][0]["params"]["modules"] = 3
    make_run(tmp_path, "grown", identifier="EXP-005", selected=grown)
    choices, errors = compatible_checkpoints(
        tmp_path, {"architecture_id": "EXP-005", "architecture": selected}
    )
    assert not errors and len(choices) == 3
    selected["nodes"][0]["params"]["max_modules"] = 2
    assert (
        compatible_checkpoints(tmp_path, {"architecture_id": "EXP-005", "architecture": selected})[
            0
        ]
        == []
    )


def test_mismatched_config_file_and_failed_run_do_not_supply_models(tmp_path):
    directory, config = make_run(tmp_path, "mismatch")
    file_config = deepcopy(config)
    file_config["architecture"] = architecture("gru")
    (directory / "config.json").write_text(json.dumps(file_config))
    make_run(tmp_path, "failed", status="FAILED")
    choices, errors = compatible_checkpoints(tmp_path, config)
    assert not choices
    assert len(errors) == 1 and "no coincide" in errors[0]


def test_external_checkpoint_symlink_never_offered(tmp_path):
    directory, config = make_run(tmp_path, "symlink")
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"external")
    path = directory / "checkpoints/last.pt"
    path.unlink()
    path.symlink_to(outside)
    choices, errors = compatible_checkpoints(tmp_path, config)
    assert not choices
    assert errors and "fuera" in errors[0]


def test_fingerprint_ignores_dict_order_but_preserves_architecture_settings():
    a = architecture()
    b = {key: a[key] for key in reversed(a)}
    assert architecture_fingerprint(a) == architecture_fingerprint(b)
    assert architecture_fingerprint(a) != architecture_fingerprint(architecture(width=16))


def test_module_import_does_not_import_torch_or_jax():
    code = (
        "import sys; import ucl.presentation; "
        "assert 'torch' not in sys.modules; assert 'jax' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
