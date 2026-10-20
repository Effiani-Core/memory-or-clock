"""Study view on artificial run summaries; no model, simulator or real study run is read.

Two studies with different names, folders and panels show that the view finds any study by
its files (see ``ucl.study_format``), not only STUDY-001.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from ucl.presentation import (
    study_catalog,
    study_checkpoint_label,
    study_manifest,
    study_task_label,
    study_task_summary,
    task_name,
)
from ucl.study_format import discover_studies, panel_order, protocol_hash

SCRIPT = Path(__file__).parents[1] / "src/ucl/dashboard.py"
TASKS = {
    "study001-A-navigation": {
        "family": "navigation", "goal": {"type": "AgentOnTileGoal", "tile": [6, 1]},
        "rules": [], "init_tiles": [[6, 1]],
    },
    "study001-B-pickup": {
        "family": "pickup", "goal": {"type": "AgentHoldGoal", "tile": [3, 1]},
        "rules": [], "init_tiles": [[3, 1], [7, 2]],
    },
    "study001-C-hold_transform": {
        "family": "hold_transform", "goal": {"type": "AgentHoldGoal", "tile": [11, 2]},
        "rules": [{"type": "AgentHoldRule", "tile": [4, 1], "prod_tile": [11, 2]}],
        "init_tiles": [[4, 1], [3, 3]],
    },
}
OTHER_TASKS = {
    "memory-X-corridor": {
        "family": "corridor_recall", "goal": {"type": "AgentOnTileGoal", "tile": [6, 2]},
        "rules": [], "init_tiles": [[6, 2]],
    },
}
RULE = {"threshold": 0.75, "consecutive": 2, "min_gain": 0.2}
LSTM, MLP = "study_001/20261017-pilot-S001-LSTM-s7-B", "study_001/20261017-pilot-S001-MLP-s7-B"
OTHER = "memoria/20261018-pilot-S002-GRU-s3-X"


def write_study(root, name, study_id, keys, tasks, sizes, **extra):
    directory = root / "configs" / name
    directory.mkdir(parents=True)
    files = {
        "protocol.json": {"study_id": study_id, "tasks": dict.fromkeys(keys, "source"), **extra},
        "tasks.json": {"tasks": tasks},
        "panels.json": {
            "task_ids": dict(zip(keys, tasks, strict=True)),
            "panels": {
                task_id: {panel: list(range(size)) for panel, size in sizes.items()}
                for task_id in tasks
            },
        },
    }
    for file_name, value in files.items():
        (directory / file_name).write_text(json.dumps(value))
    return protocol_hash(directory)


def rates(**values):
    return {key: {"success_rate": value} for key, value in values.items()}


def study_run(root, run_id, arch, digest, study_id="UCL-STUDY-001", seed=7, keys=("B",),
              acquired=64, mtime=100, others="AC"):
    directory = root / "runs" / run_id
    (directory / "checkpoints").mkdir(parents=True)
    checkpoints = {}
    for checkpoint in ("initial", *(f"{k}_{key}" for key in keys for k in ("best", "after"))):
        (directory / f"checkpoints/{checkpoint}.pt").write_bytes(b"not-torch")
        checkpoints[checkpoint] = {"path": f"checkpoints/{checkpoint}.pt", "sha256": "0" * 64}
    checkpoints["deleted"] = {"path": "checkpoints/deleted.pt", "sha256": "0" * 64}
    checkpoints["escape"] = {"path": "../../outside.pt", "sha256": "0" * 64}
    key = keys[-1]
    every = [key, *others]
    summary = {
        "study_id": study_id, "status": "FINISHED", "phase": "pilot", "protocol_hash": digest,
        "architecture_id": arch, "init_seed": seed, "task_keys": list(keys), "steps": 64,
        "curves": {key: [
            {"task_key": key, "step": step, "total_step": step, "success_rate": rate,
             "episodes": 2}
            for step, rate in ((0, 0.25), (32, 0.8), (64, 0.9))]},
        "acquisition": {key: {"step": acquired, "acquired": acquired is not None, "rule": RULE}},
        "random_baselines": {k: {"validation": {"success_rate": 0.1},
                                 "test": {"success_rate": 0.05}} for k in every},
        "matrix": {
            "initial": {"checkpoint": "checkpoints/initial.pt", "total_step": 0,
                        "test": rates(**dict.fromkeys(every, 0.2)),
                        "known": rates(**dict.fromkeys(every, 0.2))},
            f"after_{key}": {"test": rates(**{**dict.fromkeys(every, 0.4), key: 0.9}),
                             "known": rates(**{**dict.fromkeys(every, 0.5), key: 1.0})},
        },
        "checkpoints": checkpoints,
        "stages": [{"task_key": key, "start_total_step": 0}],
        "mlflow": {"url": f"http://127.0.0.1:5000/#/experiments/1/runs/{directory.name}"},
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    os.utime(directory / "summary.json", (mtime, mtime))
    (root / "runs/outside.pt").write_bytes(b"outside every run")
    return directory


def study_episode(root, folder, source_run, arch, map_number=1, success=True):
    path = root / "runs" / folder / "episodes" / f"B-test-{map_number:02d}"
    path.mkdir(parents=True)
    np.savez_compressed(path / "frames.npz", frames=np.zeros((3, 8, 8, 3), dtype=np.uint8))
    metadata = {
        "architecture_id": arch, "task_id": "study001-B-pickup", "split": "development",
        "environment": {"task_sha256": "artificial-b", "config": TASKS["study001-B-pickup"]},
        "seed": 4000 + map_number, "reward": 0.9 if success else 0.0, "success": success,
        "frames_file": "frames.npz",
        "frames_sha256": hashlib.sha256((path / "frames.npz").read_bytes()).hexdigest(),
        "steps": [{"action": 0, "reward": 0.0, "observer": {"pocket": [0, 0]}},
                  {"action": 3, "reward": 0.9, "observer": {"pocket": [3, 1]}}],
        "study": {"study_id": "UCL-STUDY-001", "source_run": source_run, "checkpoint": "after_B",
                  "task_key": "B", "panel": "test", "map_number": map_number, "init_seed": 7,
                  "mlflow": {"url": "http://127.0.0.1:5000/#/runs/x"}},
    }
    (path / "episode.json").write_text(json.dumps(metadata))
    return path / "episode.json"


@pytest.fixture
def study_lab(tmp_path, monkeypatch):
    sizes = {"train": 6, "validation": 3, "test": 5, "final": 2, "known": 2}
    digest = write_study(tmp_path, "study_001", "UCL-STUDY-001", "ABC", TASKS, sizes)
    other = write_study(tmp_path, "estudio_memoria", "UCL-STUDY-002", "X", OTHER_TASKS,
                        {"train": 4, "probe": 3, "holdout": 2},
                        sealed_panels=["holdout"])
    study_run(tmp_path, LSTM, "S001-LSTM", digest, mtime=200)
    study_run(tmp_path, MLP, "S001-MLP", digest, acquired=None, mtime=300)
    study_run(tmp_path, "study_001/20261016-pilot-S001-LSTM-s19-A", "S001-LSTM", digest, seed=19,
              keys=("A",), mtime=100, others="BC")
    study_run(tmp_path, "study_001/20261015-pilot-S001-LSTM-s31-B-old", "S001-LSTM", "f" * 64,
              seed=31, mtime=50)
    study_run(tmp_path, OTHER, "S002-GRU", other, study_id="UCL-STUDY-002", seed=3, keys=("X",),
              others="", mtime=150)
    broken = tmp_path / "runs/study_001/broken"
    broken.mkdir()
    (broken / "summary.json").write_text('{"study_id": "UCL-STUDY-001"}')
    monkeypatch.setenv("UCL_LAB_ROOT", str(tmp_path))
    return tmp_path


def test_studies_and_their_runs_are_found_by_format(study_lab):
    definitions, errors = discover_studies(study_lab)
    assert errors == []
    assert {item["study_id"]: item["name"] for item in definitions} == {
        "UCL-STUDY-002": "estudio_memoria", "UCL-STUDY-001": "study_001"}
    other = next(item for item in definitions if item["study_id"] == "UCL-STUDY-002")
    assert panel_order(other, "X") == ["train", "probe"]
    rows, errors = study_catalog(study_lab)
    assert [row["run_id"] for row in rows] == [
        MLP, LSTM, OTHER, "study_001/20261016-pilot-S001-LSTM-s19-A",
        "study_001/20261015-pilot-S001-LSTM-s31-B-old"]
    lstm = rows[1]
    assert lstm["acquired_step"] == {"B": 64} and rows[0]["acquired_step"] == {"B": None}
    assert lstm["test_after"] == 0.9 and lstm["init_seed"] == 7 and lstm["folder"].endswith("B")
    assert sorted(lstm["checkpoints"]) == ["after_B", "best_B", "initial"]
    assert lstm["definition"]["name"] == "study_001"
    assert rows[2]["definition"]["name"] == "estudio_memoria"
    assert rows[-1]["definition"] is None  # Its frozen files changed after training.
    assert len(errors) == 1 and "broken" in errors[0]


def test_broken_definitions_are_reported_not_hidden(study_lab):
    directory = study_lab / "configs/roto"
    directory.mkdir()
    for name in ("protocol.json", "tasks.json", "panels.json"):
        (directory / name).write_text("{}")
    (study_lab / "configs/solo_protocolo").mkdir()
    (study_lab / "configs/solo_protocolo/protocol.json").write_text("{}")
    definitions, errors = discover_studies(study_lab)
    assert len(definitions) == 2 and len(errors) == 1 and "roto" in errors[0]


def test_study_tasks_are_worded_for_the_observer(study_lab):
    definitions, _ = discover_studies(study_lab)
    manifest = study_manifest(next(d for d in definitions if d["name"] == "study_001"))
    assert list(manifest) == ["A", "B", "C"]
    assert manifest["B"]["panel_sizes"]["test"] == 5
    assert study_task_label("B", manifest["B"]["task"]) == "B · Recoger un objeto"
    assert study_task_label("X", OTHER_TASKS["memory-X-corridor"]) == "X · Corridor recall"
    b = dict(study_task_summary(manifest["B"]["task"]))
    assert b["Meta"] == "Tener en la mano la pelota de color rojo."
    assert b["Distractor"].startswith("la llave de color verde")
    c = dict(study_task_summary(manifest["C"]["task"]))
    assert c["Meta"] == "Tener en la mano el hexágono de color verde."
    assert c["Regla"] == (
        "Al recoger el cuadrado de color rojo, se convierte en el hexágono de color verde.")
    assert c["En la práctica"] == "Recoger el cuadrado de color rojo."
    assert c["Distractor"].startswith("la pelota de color azul")
    assert task_name("study001-C-hold_transform").endswith("· tarea C")
    assert study_checkpoint_label("after_B") == "Al terminar la tarea B"
    assert study_checkpoint_label("best_A") == "Mejor en validación · tarea A"
    assert study_checkpoint_label("initial") == "Al empezar · sin entrenar"


def test_curve_matrix_and_rule_text_come_from_the_summary(study_lab):
    from ucl.dashboard import (
        acquisition_rule_text,
        checkpoint_order,
        learning_curve,
        matrix_panels,
        study_matrix,
    )

    rows, _ = study_catalog(study_lab)
    summary = rows[1]["summary"]
    chart, points = learning_curve(summary)
    assert chart is not None and [p["Éxito"] for p in points] == [25.0, 80.0, 90.0]
    assert matrix_panels(summary) == ["test", "known"]
    table, keys = study_matrix(summary, "test")
    assert keys == ["B", "A", "C"]  # The trained task first, then transfer columns.
    assert [row["Momento"] for row in table] == [
        "Al empezar · sin entrenar", "Al terminar la tarea B", "Agente al azar"]
    assert table[1]["Tarea B"] == 90.0 and table[2]["Tarea C"] == 5.0
    assert [row["Momento"] for row in study_matrix(summary, "known")[0]][-1] != "Agente al azar"
    text = acquisition_rule_text(summary)
    assert "2 evaluaciones seguidas" in text and "75%" in text and "20 puntos" in text
    assert checkpoint_order(["initial", "best_B", "after_B"], ["B"]) == [
        "after_B", "best_B", "initial"]
    assert learning_curve({"task_keys": ["A"], "curves": {}}) == (None, [])
    assert matrix_panels({"matrix": {}}) == []


def test_study_view_opens_first_on_the_newest_run(study_lab):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(SCRIPT), default_timeout=20).run()
    assert not app.exception
    assert app.radio(key="lab_view").value == "Estudio"
    assert app.selectbox(key="study_id").options == ["UCL-STUDY-001", "UCL-STUDY-002"]
    assert app.session_state["study_id"] == "UCL-STUDY-001"
    assert app.session_state["study_arch"] == "S001-MLP"
    assert app.session_state["study_task"] == "B"
    assert app.selectbox(key="study_checkpoint").value == "after_B"
    assert len(app.selectbox(key="study_checkpoint").options) == 3
    assert any("no adquirida en 64" in item.value for item in app.markdown)
    assert any("Meta:** Tener en la mano la pelota" in item.value for item in app.markdown)
    assert app.dataframe
    assert app.checkbox(key="study_compare").value is True
    assert app.multiselect(key="study_maps_B_test").value == [1, 2, 3, 4]
    assert not app.button(key="study_record").disabled
    assert any("Sellado hasta la evaluación final del estudio: Final" in item.value
               for item in app.caption)
    panels = app.segmented_control(key="study_panel").options
    assert "Final" not in panels and {"Validación", "Test", "Mapas ya vistos"} <= set(panels)


def test_another_study_shows_its_own_tasks_and_sealed_panels(study_lab):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(SCRIPT), default_timeout=20)
    app.query_params["view"] = "estudio"
    app.query_params["run"] = OTHER
    app.run()
    assert not app.exception
    assert app.session_state["study_id"] == "UCL-STUDY-002"
    assert app.session_state["study_arch"] == "S002-GRU"
    assert app.session_state["study_watch_task"] == "X"
    assert app.session_state["study_panel"] == "train"
    assert app.multiselect(key="study_maps_X_train").value == [1, 2, 3, 4]
    assert any("Sellado hasta la evaluación final del estudio: Holdout" in item.value
               for item in app.caption)
    assert list(app.segmented_control(key="study_panel").options) == ["Entrenamiento", "Probe"]
    app.selectbox(key="study_id").select("UCL-STUDY-001").run()
    assert not app.exception
    assert app.session_state["study_arch"] in {"S001-LSTM", "S001-MLP"}


def test_a_run_whose_protocol_changed_can_be_seen_but_not_recorded(study_lab):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(SCRIPT), default_timeout=20)
    app.query_params["run"] = "20261015-pilot-S001-LSTM-s31-B-old"  # An old bare-folder link.
    app.run()
    assert not app.exception
    assert app.session_state["study_seed"] == 31
    assert any("no se pueden grabar episodios" in item.value for item in app.warning)
    assert not [button for button in app.button if button.key == "study_record"]


def test_deep_link_selects_the_run_and_its_episode_opens_in_simulation(study_lab):
    from streamlit.testing.v1 import AppTest

    mine = study_episode(study_lab, "ui-a-study-00", LSTM, "S001-LSTM")
    # Recordings made before runs were named by path carry the bare folder name.
    partner = study_episode(study_lab, "ui-a-study-01", Path(MLP).name, "S001-MLP", success=False)
    app = AppTest.from_file(str(SCRIPT), default_timeout=20)
    app.query_params["view"] = "estudio"
    app.query_params["run"] = LSTM
    app.run()
    assert not app.exception
    assert app.session_state["study_arch"] == "S001-LSTM"
    assert any("adquirida a 64 interacciones" in item.value for item in app.markdown)
    from ucl.dashboard import episode_key

    button = app.button(key=episode_key(str(mine)))
    assert button.label == "Ver con comparación"
    button.click().run()
    assert not app.exception
    assert app.radio(key="lab_view").value == "Ver un agente"
    assert app.session_state["episode_a"] == str(mine)
    assert app.session_state["episode_b"] == str(partner)
    assert app.session_state["compare_agents"] is True
    assert any("Meta:** Tener en la mano la pelota" in item.value for item in app.markdown)
    app.button(key="episode_study").click().run()
    assert not app.exception
    assert app.radio(key="lab_view").value == "Estudio"
    assert app.session_state["study_arch"] == "S001-LSTM"


def test_study_view_without_runs_explains_how_to_start(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("UCL_LAB_ROOT", str(tmp_path))
    app = AppTest.from_file(str(SCRIPT), default_timeout=20)
    app.query_params["view"] = "estudio"
    app.run()
    assert not app.exception
    assert any("study-pilot" in item.value and "formato" in item.value for item in app.info)


def test_study_job_records_both_runs_and_retries_only_on_request(study_lab, monkeypatch):
    from ucl.dashboard import EvaluationJobs

    commands, results = [], iter([0, 1, 0, 0])

    class Child:
        pid = 4321

        def __init__(self, command, **kwargs):
            commands.append(command)
            assert kwargs["cwd"] == study_lab and kwargs["env"]["JAX_PLATFORMS"] == "cpu"

        def wait(self):
            return next(results)

    monkeypatch.setattr("ucl.dashboard.subprocess.Popen", Child)
    manager = EvaluationJobs(study_lab)
    runs = [LSTM, MLP]
    with pytest.raises(ValueError, match="sealed"):
        manager.start_study(runs, "after_B", "B", "final", [1])
    with pytest.raises(ValueError, match="Not a study run"):
        manager.start_study(["../elsewhere"], "after_B", "B", "test", [1])
    with pytest.raises(ValueError, match="same study protocol"):
        manager.start_study([LSTM, OTHER], "after_B", "B", "test", [1])
    with pytest.raises(ValueError, match="protocol changed"):
        manager.start_study(["study_001/20261015-pilot-S001-LSTM-s31-B-old"], "after_B", "B",
                            "test", [1])
    assert commands == [] and manager.status() is None
    job = manager.start_study(runs, "after_B", "B", "test", [1, 3])
    manager.worker.join(5)
    status = manager.status()
    assert status["state"] == "failed" and len(commands) == 2
    assert commands[0][commands[0].index("--run") + 1] == runs[0]
    assert commands[1][commands[1].index("--run") + 1] == runs[1]
    assert commands[0][commands[0].index("--maps") + 1] == "1,3"
    assert all(Path(output).parent == study_lab / "runs" for output in job["outputs"])
    assert job["outputs"][0].endswith("-study-00")
    retried = manager.retry()
    manager.worker.join(5)
    assert retried["id"] != job["id"] and retried["request"] == job["request"]
    assert manager.status()["state"] == "finished" and len(commands) == 4
    with pytest.raises(RuntimeError, match="no failed run"):
        manager.retry()


def test_a_deep_link_opens_the_exact_run_even_when_its_combination_repeats(study_lab):
    from streamlit.testing.v1 import AppTest

    rerun = "study_001/20261016-pilot-S001-LSTM-s7-B-rerun"
    digest = protocol_hash(study_lab / "configs/study_001")
    study_run(study_lab, rerun, "S001-LSTM", digest, acquired=32, mtime=120)
    app = AppTest.from_file(str(SCRIPT), default_timeout=20)
    app.query_params["run"] = rerun
    app.run()
    assert not app.exception
    assert app.session_state["study_run_id"] == rerun
    assert len(app.selectbox(key="study_run_id").options) == 2
    assert any("adquirida a 32 interacciones" in item.value for item in app.markdown)
    app.selectbox(key="study_run_id").select(LSTM).run()
    assert any("adquirida a 64 interacciones" in item.value for item in app.markdown)


def test_coming_back_to_the_view_keeps_the_chosen_run(study_lab):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(SCRIPT), default_timeout=20)
    app.query_params["run"] = OTHER
    app.run()
    app.radio(key="lab_view").set_value("Resultados").run()
    app.radio(key="lab_view").set_value("Estudio").run()
    assert not app.exception
    assert app.session_state["study_arch"] == "S002-GRU"


def test_unclear_links_say_so_instead_of_opening_another_run(study_lab):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(SCRIPT), default_timeout=20)
    app.query_params["run"] = "study_001/does-not-exist"
    app.run()
    assert not app.exception
    assert any("No se encontró la ejecución" in item.value for item in app.warning)


def test_a_summary_with_wrong_types_goes_to_diagnostics_not_a_crash(study_lab):
    from streamlit.testing.v1 import AppTest

    path = study_lab / "runs" / MLP / "summary.json"
    summary = json.loads(path.read_text())
    summary["phase"] = ["pilot"]
    path.write_text(json.dumps(summary))
    rows, errors = study_catalog(study_lab)
    assert MLP not in [row["run_id"] for row in rows]
    assert any("phase" in error for error in errors)
    app = AppTest.from_file(str(SCRIPT), default_timeout=20).run()
    assert not app.exception and app.session_state["study_arch"] == "S001-LSTM"


def test_clicking_record_launches_both_agents_in_the_background(study_lab, monkeypatch):
    import subprocess
    import time

    from streamlit.testing.v1 import AppTest

    commands = []

    class Child:
        pid = 99

        def __init__(self, command, **kwargs):
            commands.append(command)

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", Child)
    app = AppTest.from_file(str(SCRIPT), default_timeout=20).run()
    app.segmented_control(key="study_watch_task").set_value("C").run()
    app.button(key="study_record").click().run()
    assert not app.exception
    for _ in range(50):
        if len(commands) == 2:
            break
        time.sleep(0.1)
    runs = [command[command.index("--run") + 1] for command in commands]
    assert runs == [MLP, LSTM]
    command = commands[0]
    assert command[command.index("--task") + 1] == "C"
    assert command[command.index("--panel") + 1] == "test"
    assert command[command.index("--maps") + 1] == "1,2,3,4"
    assert command[command.index("--checkpoint") + 1] == "after_B"


def test_recording_results_are_reported_per_agent(tmp_path):
    from ucl.dashboard import recording_results

    for folder, arch, outcomes in (("a", "S001-LSTM", [True, True, False]),
                                   ("b", "S001-MLP", [False, False, True])):
        for index, success in enumerate(outcomes):
            path = tmp_path / folder / "episodes" / f"B-test-{index:02d}"
            path.mkdir(parents=True)
            (path / "episode.json").write_text(
                json.dumps({"architecture_id": arch, "success": success}))
    assert recording_results([tmp_path / "a", tmp_path / "b"]) == (
        "Memoria LSTM: 2 de 3 con éxito · Sin memoria (control): 1 de 3 con éxito.")


def test_rules_that_join_objects_are_not_called_distractors():
    adjacency = {
        "goal": {"type": "AgentHoldGoal", "tile": [12, 3]},
        "rules": [{"type": "TileNearRule", "tile_a": [3, 1], "tile_b": [5, 2],
                   "prod_tile": [12, 3]}],
        "init_tiles": [[3, 1], [5, 2], [7, 3]],
    }
    lines = dict(study_task_summary(adjacency))
    assert lines["Regla"] == (
        "Al juntar la pelota de color rojo con la pirámide de color verde, se convierten en "
        "la estrella de color azul.")
    assert lines["Distractor"] == "la llave de color azul (no sirve para la meta)."
    unknown = {**adjacency, "rules": [{"type": "SomeFutureRule", "tile": [3, 1]}]}
    assert "Distractor" not in dict(study_task_summary(unknown))
