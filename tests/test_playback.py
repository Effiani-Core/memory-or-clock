"""Replay/UI checks use artificial fixtures, never scientific performance claims."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from ucl.playback import (
    comparable_recordings,
    confined_path,
    discover_recordings,
    load_recording,
    next_playback_step,
    read_metrics,
    step_details,
    synchronized_indices,
)


def episode(root, name="lstm", steps=3):
    path = root / "runs" / name / "episodes" / "navigation-7"
    path.mkdir(parents=True)
    frames = np.stack([np.full((8, 8, 3), i, dtype=np.uint8) for i in range(steps + 1)])
    np.savez_compressed(path / "frames.npz", frames=frames)
    metadata = {
        "architecture_id": name,
        "task_id": "development-navigation",
        "split": "development",
        "environment": {"task_sha256": hashlib.sha256(b"synthetic-navigation-task").hexdigest()},
        "seed": 7,
        "reward": 0.5,
        "success": True,
        "frames_file": "frames.npz",
        "frames_sha256": hashlib.sha256((path / "frames.npz").read_bytes()).hexdigest(),
        "steps": [
            {
                "action": i % 6,
                "reward": 0.5 if i == steps - 1 else 0,
                "terminated": i == steps - 1,
                "truncated": False,
                "signals": {"hidden_norm": i + 0.1},
            }
            for i in range(steps)
        ],
    }
    (path / "episode.json").write_text(json.dumps(metadata))
    return path / "episode.json", metadata


def test_replay_verifies_pixels_and_real_signals(tmp_path):
    path, _ = episode(tmp_path)
    recording = load_recording(path, tmp_path)
    assert recording.length == 4
    assert not recording.frames.flags.writeable
    assert np.all(recording.frames[2] == 2)
    assert step_details(recording, 0)["signals"] == {}
    assert step_details(recording, 2)["signals"] == {"hidden_norm": 1.1}
    assert step_details(recording, 100)["finished"]


def test_unequal_episodes_synchronize_by_step():
    assert synchronized_indices([4, 7], 0) == [0, 0]
    assert synchronized_indices([4, 7], 2) == [2, 2]
    assert synchronized_indices([4, 7], 5) == [3, 5]
    assert synchronized_indices([4, 7], 99) == [3, 6]
    with pytest.raises(ValueError):
        synchronized_indices([0, 3], 1)
    with pytest.raises(ValueError):
        synchronized_indices([4], -1)


def test_play_pause_and_speed():
    assert next_playback_step(1, 10, False, 10, 4) == (1, False)
    assert next_playback_step(1, 10, True, 1, 1) == (3, True)
    assert next_playback_step(1, 10, True, 1, 2) == (5, True)
    assert next_playback_step(9, 10, True, 10, 4) == (10, False)
    with pytest.raises(ValueError):
        next_playback_step(0, 10, True, 1, 0)


def test_tampered_recording_is_rejected(tmp_path):
    path, _ = episode(tmp_path)
    (path.parent / "frames.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_recording(path, tmp_path)


def test_wrong_frame_count_rejected(tmp_path):
    path, metadata = episode(tmp_path)
    metadata["steps"].pop()
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="initial frame"):
        load_recording(path, tmp_path)


def test_path_traversal_is_rejected(tmp_path):
    path, metadata = episode(tmp_path)
    metadata["frames_file"] = "../../other.npz"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="outside"):
        load_recording(path, tmp_path)
    with pytest.raises(ValueError, match="outside"):
        confined_path(tmp_path.parent / "external", tmp_path)


def test_object_archive_is_not_deserialized(tmp_path):
    path, metadata = episode(tmp_path)
    np.savez(path.parent / "frames.npz", frames=np.array([object()], dtype=object))
    metadata["frames_sha256"] = hashlib.sha256(
        (path.parent / "frames.npz").read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Object arrays"):
        load_recording(path, tmp_path)


def test_discovery_and_partial_metric_writer(tmp_path):
    episode(tmp_path)
    episode(tmp_path, "gru", 5)
    records, errors = discover_recordings(tmp_path)
    assert len(records) == 2 and errors == []
    assert all(record["split"] == "development" for record in records)
    assert all(
        record["task_sha256"] == hashlib.sha256(b"synthetic-navigation-task").hexdigest()
        for record in records
    )
    assert comparable_recordings(*records)
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step":1,"loss":2}\n{"step":2,"loss":1}\n{"step":')
    assert read_metrics(path) == [{"step": 1, "loss": 2}, {"step": 2, "loss": 1}]


@pytest.mark.parametrize(
    "changed",
    [
        {"task_id": "another-task"},
        {"seed": 8},
        {"task_sha256": hashlib.sha256(b"changed-rules").hexdigest()},
        {"task_sha256": None},
        {"task_sha256": ""},
        {"task_sha256": "   "},
    ],
)
def test_comparison_rejects_changed_scenario_or_unknown_hash(changed):
    original = {
        "task_id": "development-navigation",
        "seed": 7,
        "task_sha256": hashlib.sha256(b"synthetic-navigation-task").hexdigest(),
    }
    other = {**original, **changed}
    assert not comparable_recordings(original, other)
    assert not comparable_recordings(other, original)


@pytest.mark.parametrize("missing", ["task_id", "seed", "task_sha256"])
def test_comparison_requires_provenance_even_when_both_records_lack_it(missing):
    record = {"task_id": "task", "seed": 7, "task_sha256": "synthetic-hash"}
    record.pop(missing)
    assert not comparable_recordings(record, record.copy())


def test_discovery_preserves_recorded_split_and_does_not_invent_hash(tmp_path):
    path, metadata = episode(tmp_path)
    metadata["task_id"] = "custom-task-without-split-in-name"
    metadata["split"] = "validation"
    metadata.pop("environment")
    path.write_text(json.dumps(metadata))
    records, errors = discover_recordings(tmp_path)
    assert errors == []
    assert records[0]["split"] == "validation"
    assert records[0]["task_sha256"] is None
    assert not comparable_recordings(records[0], records[0].copy())


def test_replay_does_not_import_training_or_simulator(tmp_path):
    path, _ = episode(tmp_path)
    code = (
        "import sys; from ucl.playback import load_recording; "
        f"r=load_recording({str(path)!r}, {str(tmp_path)!r}); "
        "assert len(r.frames)==4; "
        "assert not any(x in sys.modules for x in ('torch','jax','xminigrid','ucl.envs'))"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def ui_lab(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/tasks.json").write_text(
        json.dumps(
            {
                "tasks": {"development-navigation": {"adapter": "xland", "split": "development"}},
                "curricula": {
                    "return_to_previous": {"stages": [{"task_id": "development-navigation"}]}
                },
            }
        )
    )
    for architecture in ("EXP-002", "EXP-003"):
        (tmp_path / "configs" / f"{architecture}.json").write_text(
            json.dumps({"architecture_id": architecture, "architecture": {"kind": "graph"}})
        )
    episode(tmp_path, "lstm")
    episode(tmp_path, "gru", 5)
    (tmp_path / "runs/lstm/summary.json").write_text(json.dumps({"architecture_id": "lstm"}))
    (tmp_path / "runs/lstm/metrics.jsonl").write_text('{"step":1,"loss":0.5}\n')
    monkeypatch.setenv("UCL_LAB_ROOT", str(tmp_path))
    return tmp_path


def test_dashboard_selectors_and_step_controls(ui_lab):
    from streamlit.testing.v1 import AppTest

    script = Path(__file__).parents[1] / "src/ucl/dashboard.py"
    app = AppTest.from_file(str(script), default_timeout=15).run()
    assert not app.exception
    assert app.radio(key="lab_view").value == "Ver un agente"
    assert "architecture" not in [item.key for item in app.selectbox]
    app.radio(key="lab_view").set_value("Probar un modelo").run()
    assert app.selectbox(key="environment").value == "xland"
    app.selectbox(key="architecture").select(ui_lab / "configs/EXP-003.json").run()
    assert not app.exception
    app.radio(key="evaluation_mode").set_value("Un currículo").run()
    assert app.selectbox(key="curriculum").value == "return_to_previous"
    app.radio(key="lab_view").set_value("Ver un agente").run()
    app.selectbox(key="watch_model").select("gru").run()
    app.checkbox(key="compare_agents").check().run()
    second = str(ui_lab / "runs/lstm/episodes/navigation-7/episode.json")
    app.selectbox(key="episode_b").select(second).run()
    app.button(key="one_step").click().run()
    assert not app.exception
    assert app.session_state["replay_position"] == 1
    assert app.session_state["playback_playing"] is False
    app.select_slider(key="speed").set_value(2.0).run()
    assert app.select_slider(key="speed").value == 2.0
    app.slider(key="playback_step").set_value(2).run()
    assert app.session_state["replay_position"] == 2
    app.button(key="reset_playback").click().run()
    assert app.session_state["replay_position"] == 0
    app.button(key="play_pause").click().run()
    assert app.session_state["playback_playing"] is True
    app.button(key="play_pause").click().run()
    assert app.session_state["playback_playing"] is False
    assert not app.exception


def test_job_is_async_serialized_and_uses_cli(ui_lab, monkeypatch):
    from ucl.dashboard import EvaluationJobs

    entered, release = threading.Event(), threading.Event()
    commands = []

    class Child:
        pid = 123456

        def __init__(self, command, **kwargs):
            commands.append(command)
            assert kwargs["env"]["JAX_PLATFORMS"] == "cpu"
            assert kwargs["cwd"] == ui_lab
            entered.set()

        def wait(self):
            assert release.wait(5)
            return 0

    monkeypatch.setattr("ucl.dashboard.subprocess.Popen", Child)
    manager = EvaluationJobs(ui_lab)
    manager.start(
        ui_lab / "configs/EXP-002.json",
        ["development-navigation"],
        7,
        curriculum="return_to_previous",
    )
    assert entered.wait(3)
    try:
        assert manager.status()["state"] == "running"
        with pytest.raises(RuntimeError, match="already running"):
            manager.start(ui_lab / "configs/EXP-003.json", ["development-navigation"], 7)
        assert "--curriculum" in commands[0]
        assert "--task" not in commands[0]
    finally:
        release.set()
        manager.worker.join(5)
    assert manager.status()["state"] == "finished"
    assert manager.status()["completed"] == 1


def test_discovery_ignores_mlflow_artifact_copies(tmp_path):
    from ucl.playback import run_files

    real, _ = episode(tmp_path, "original")
    episode(tmp_path, "mlartifacts/run-id/artifacts")
    records, errors = discover_recordings(tmp_path)
    assert errors == []
    assert [record["path"] for record in records] == [str(real)]
    assert run_files(tmp_path, "**/episode.json") == [real]


def test_architecture_discovery_includes_custom_and_marks_unavailable(tmp_path):
    from ucl.playback import discover_architecture_configs

    directory = tmp_path / "configs"
    directory.mkdir()
    (directory / "tasks.json").write_text('{"tasks": {}}')
    custom = directory / "CUSTOM-EXAMPLE.json"
    custom.write_text(json.dumps({"architecture_id": "custom", "architecture": {"kind": "graph"}}))
    (directory / "OPTIONAL.json").write_text(
        json.dumps(
            {
                "architecture_id": "larger",
                "architecture": {"kind": "frozen_llm_baseline"},
                "availability": "unavailable",
                "unavailable_reason": "No local model installed",
            }
        )
    )
    available, unavailable = discover_architecture_configs(tmp_path)
    assert available == [custom]
    assert unavailable == [{"name": "larger", "reason": "No local model installed"}]


def test_dashboard_lists_custom_configuration_and_excludes_unavailable(ui_lab):
    from streamlit.testing.v1 import AppTest

    custom = ui_lab / "configs/CUSTOM-EXAMPLE.json"
    custom.write_text(json.dumps({"architecture_id": "custom", "architecture": {"kind": "graph"}}))
    (ui_lab / "configs/OPTIONAL.json").write_text(
        json.dumps(
            {
                "architecture_id": "larger",
                "architecture": {"kind": "frozen_llm_baseline"},
                "availability": "unavailable",
                "unavailable_reason": "Not installed",
            }
        )
    )
    script = Path(__file__).parents[1] / "src/ucl/dashboard.py"
    app = AppTest.from_file(str(script), default_timeout=15).run()
    assert not app.exception
    app.radio(key="lab_view").set_value("Probar un modelo").run()
    options = app.selectbox(key="architecture").options
    assert len(options) == 3
    from ucl.presentation import architecture_name

    assert (
        "custom · "
        + architecture_name(
            "custom", {"architecture_id": "custom", "architecture": {"kind": "graph"}}
        )
        in options
    )
    assert not any("OPTIONAL" in option for option in options)
    app.selectbox(key="architecture").select(custom).run()
    assert app.selectbox(key="architecture").value == custom
    assert not app.exception
