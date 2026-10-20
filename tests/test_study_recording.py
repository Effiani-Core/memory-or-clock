"""Study replays on tiny hand-built 7x7 panels; never reads the real study panels or runs.

The tiny study is written to ``configs/study_test`` (not ``study_001``) so these tests also show
that a study is found by its files, never by a fixed folder name.
"""

import copy
import fcntl
import json
import shutil

import pytest
from test_study import ARCHITECTURE, SOURCES, tiny_study

from ucl.cli import main as cli_main
from ucl.playback import discover_recordings, load_recording
from ucl.presentation import run_catalog, study_catalog
from ucl.study import run_study
from ucl.study_format import panel_order, read_definition, resolve_study_run
from ucl.study_recording import check_request, record_study_episodes

RUN = "study_001/run-a"


def write_definition(root, name, study, **protocol_extra):
    directory = root / "configs" / name
    directory.mkdir(parents=True)
    protocol = {**copy.deepcopy(study["protocol"]), "tasks": dict(SOURCES), **protocol_extra}
    ids = list(study["tasks"])
    manifest = {"schema_version": 1, "tasks": study["tasks"],
                "splits": {"development": ids, "validation": [], "heldout": []}}
    files = {"protocol.json": protocol, "tasks.json": manifest,
             "panels.json": {"task_ids": study["task_ids"], "panels": study["panels"]}}
    for file_name, value in files.items():
        (directory / file_name).write_text(json.dumps(value, indent=2))
    return read_definition(directory, validate_tasks=True)


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    """One finished single-task run, trained for a few steps against configs/study_test."""
    root = tmp_path_factory.mktemp("lab")
    definition = write_definition(root, "study_test", tiny_study())
    study = {key: definition[key] for key in ("protocol", "tasks", "panels", "task_ids",
                                              "protocol_hash")}
    run_study(study, ARCHITECTURE, 7, ["A"], steps=16, eval_every=16,
              output=root / "runs" / RUN, root=root)
    return root, definition


def test_recording_reproduces_the_study_panel_evaluation(lab):
    root, definition = lab
    run = root / "runs" / RUN
    before = (run / "summary.json").read_bytes()
    output = root / "runs/ui-reproduce-study-00"
    result = record_study_episodes(root, RUN, "after_A", "A", "test", [1, 2], output)
    summary = json.loads(before)
    seeds = definition["panels"][definition["task_ids"]["A"]]["test"]
    assert result["kind"] == "study_recording" and result["parameters_updated"] is False
    assert result["source_run"] == RUN
    assert [episode["seed"] for episode in result["episodes"]] == seeds
    successes = [episode["success"] for episode in result["episodes"]]
    assert sum(successes) / 2 == summary["matrix"]["after_A"]["test"]["A"]["success_rate"]
    saved = json.loads((output / "episodes/A-test-01/episode.json").read_text())
    context = saved["study"]
    assert context["source_run"] == RUN and context["map_number"] == 1
    assert context["study_definition"] == "study_test" and context["panel"] == "test"
    assert context["checkpoint_sha256"] == summary["checkpoints"]["after_A"]["sha256"]
    assert context["protocol_hash"] == definition["protocol_hash"] == summary["protocol_hash"]
    assert all(isinstance(step["observer"]["pocket"], list) for step in saved["steps"])
    # The study's panel evaluation samples actions from the map seed itself.
    assert saved["sampling_seed"] == saved["seed"] == seeds[0]
    measured = summary["matrix"]["after_A"]["test"]["A"]
    assert sum(episode["actions"] for episode in result["episodes"]) == measured[
        "environment_steps"]
    assert sum(episode["reward"] for episode in result["episodes"]) / 2 == pytest.approx(
        measured["reward_mean"])
    assert load_recording(output / "episodes/A-test-01/episode.json", root).length >= 1
    assert (run / "summary.json").read_bytes() == before
    assert json.loads((output / "recording.json").read_text())["episodes"][1]["map_number"] == 2


def test_recordings_are_discovered_but_never_listed_as_runs(lab):
    root, definition = lab
    output = root / "runs/ui-discovery-study-00"
    record_study_episodes(root, RUN, "initial", "B", "validation", [1], output)
    records, errors = discover_recordings(root)
    mine = [record for record in records if str(output) in record["path"]]
    assert errors == [] and len(mine) == 1
    assert mine[0]["study"]["task_key"] == "B" and mine[0]["study"]["checkpoint"] == "initial"
    rows, catalog_errors = run_catalog(root)
    assert rows == [] and catalog_errors == []
    studies, study_errors = study_catalog(root)
    assert study_errors == [] and [row["run_id"] for row in studies] == [RUN]
    assert studies[0]["definition"]["protocol_hash"] == definition["protocol_hash"]
    assert studies[0]["study_id"] == "UCL-STUDY-001-TEST" and studies[0]["folder"] == "run-a"


def test_a_study_run_is_found_anywhere_under_runs_and_by_its_old_bare_name(lab):
    root, _ = lab
    elsewhere = root / "runs/other_place/run-b"
    shutil.copytree(root / "runs" / RUN, elsewhere)
    try:
        assert resolve_study_run(root, "other_place/run-b")[0] == elsewhere.resolve()
        assert resolve_study_run(root, "run-b")[0] == elsewhere.resolve()
        assert resolve_study_run(root, "run-a")[0] == (root / "runs" / RUN).resolve()
        result = record_study_episodes(root, "other_place/run-b", "after_A", "C", "known", [1],
                                       root / "runs/ui-elsewhere-study-00")
        assert result["source_run"] == "other_place/run-b"
        assert {row["run_id"] for row in study_catalog(root)[0]} == {RUN, "other_place/run-b"}
    finally:
        shutil.rmtree(elsewhere)


@pytest.mark.parametrize(
    ("run_id", "checkpoint", "task", "panel", "maps", "match"),
    [
        (RUN, "after_A", "A", "final", [1], "sealed"),
        (RUN, "after_A", "A", "test", [3], "between 1 and 2"),
        (RUN, "after_A", "A", "test", [1, 1], "distinct"),
        (RUN, "after_A", "A", "test", list(range(1, 18)), "distinct"),
        (RUN, "after_A", "A", "test", ["uno"], "whole numbers"),
        (RUN, "after_A", "Z", "test", [1], "Unknown study task"),
        (RUN, "after_A", "A", "nope", [1], "Unknown panel"),
        (RUN, "after_B", "A", "test", [1], "not recorded"),
        ("../run-a", "after_A", "A", "test", [1], "Not a study run"),
        (f"{RUN}/checkpoints", "after_A", "A", "test", [1], "Not a study run"),
        ("", "after_A", "A", "test", [1], "Not a study run"),
    ],
)
def test_invalid_requests_fail_before_recording(lab, run_id, checkpoint, task, panel, maps,
                                                match):
    root, _ = lab
    output = root / "runs/ui-refused"
    with pytest.raises((ValueError, OSError), match=match):
        record_study_episodes(root, run_id, checkpoint, task, panel, maps, output)
    assert not output.exists()


def test_changed_checkpoint_protocol_and_outside_output_are_refused(lab):
    root, _ = lab
    copies = {}
    for name, change in {
        "run-changed-checkpoint": ("checkpoints", "0" * 64),
        "run-changed-protocol": ("protocol_hash", "f" * 64),
        "run-other-study": ("study_id", "UCL-STUDY-UNKNOWN"),
    }.items():
        directory = root / "runs/study_001" / name
        shutil.copytree(root / "runs" / RUN, directory)
        summary = json.loads((directory / "summary.json").read_text())
        if change[0] == "checkpoints":
            summary["checkpoints"]["after_A"]["sha256"] = change[1]
        else:
            summary[change[0]] = change[1]
        (directory / "summary.json").write_text(json.dumps(summary))
        copies[name] = directory
    try:
        cases = {"run-changed-checkpoint": "changed after the run",
                 "run-changed-protocol": "protocol changed",
                 "run-other-study": "No study definition"}
        for name, match in cases.items():
            with pytest.raises(ValueError, match=match):
                record_study_episodes(root, f"study_001/{name}", "after_A", "A", "test", [1],
                                      root / "runs/ui-changed")
        with pytest.raises(ValueError, match="inside runs/"):
            record_study_episodes(root, RUN, "after_A", "A", "test", [1], root / "elsewhere")
        with pytest.raises(ValueError, match="inside runs/"):
            record_study_episodes(root, RUN, "after_A", "A", "test", [1], root / "runs")
        rows = {row["run_id"]: row for row in study_catalog(root)[0]}
        assert rows["study_001/run-changed-protocol"]["definition"] is None
        assert rows["study_001/run-other-study"]["definition"] is None
    finally:
        for directory in copies.values():
            shutil.rmtree(directory)
    assert not (root / "runs/ui-changed").exists() and not (root / "elsewhere").exists()


def test_busy_lab_is_refused_instead_of_waiting(lab):
    root, _ = lab
    with (root / "runs/.execution.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another laboratory execution"):
            record_study_episodes(root, RUN, "after_A", "A", "test", [1], root / "runs/ui-busy")
    assert not (root / "runs/ui-busy").exists()


def test_sealed_panels_come_from_the_protocol(lab):
    _, definition = lab
    assert panel_order(definition, "A") == ["validation", "test", "known", "train"]
    assert check_request(definition, "A", "test", ["2", 1])[2] == [2, 1]
    custom = {**definition, "sealed_panels": ["final", "test"]}
    assert panel_order(custom, "A") == ["validation", "known", "train"]
    with pytest.raises(ValueError, match="sealed"):
        check_request(custom, "A", "test", [1])
    opened = {**definition, "sealed_panels": []}
    assert "final" in panel_order(opened, "A")


def test_cli_passes_any_task_and_panel_to_the_recorder(monkeypatch, capsys):
    calls = []

    def recorder(*args):
        calls.append(args)
        return {"status": "FINISHED"}

    monkeypatch.setattr("ucl.study_recording.record_study_episodes", recorder)
    assert cli_main(["study-record", "--run", "study_002/x", "--checkpoint", "after_X",
                     "--task", "X", "--panel", "holdout", "--maps", "1, 3",
                     "--output", "runs/ui-x"]) == 0
    root, run, checkpoint, task, panel, maps, output = calls[0]
    assert (run, checkpoint, task, panel, maps) == ("study_002/x", "after_X", "X", "holdout",
                                                    ["1", "3"])
    assert output == root / "runs/ui-x" and '"FINISHED"' in capsys.readouterr().out


@pytest.mark.parametrize("inside", [RUN + "/replays", RUN + "/checkpoints/replays",
                                    "mlruns-isolated/replays", "mlartifacts/x/artifacts/y",
                                    "ui_jobs/replays", "ui-old-recording/again"])
def test_recordings_never_go_inside_a_run_a_recording_or_an_mlflow_copy(lab, inside):
    root, _ = lab
    (root / "runs/ui-old-recording").mkdir(parents=True, exist_ok=True)
    (root / "runs/ui-old-recording/recording.json").write_text("{}")
    before = sorted(path.relative_to(root) for path in (root / "runs" / RUN).rglob("*"))
    with pytest.raises(ValueError, match="cannot go inside"):
        record_study_episodes(root, RUN, "after_A", "A", "test", [1], root / "runs" / inside)
    assert not (root / "runs" / inside).exists()
    assert sorted(path.relative_to(root) for path in (root / "runs" / RUN).rglob("*")) == before


@pytest.mark.parametrize("copy", ["mlartifacts/abc/artifacts", "mlruns-isolated/abc/run-a"])
def test_mlflow_copies_of_a_run_are_never_resolved_as_runs(lab, copy):
    root, _ = lab
    mirror = root / "runs" / copy
    shutil.copytree(root / "runs" / RUN, mirror)
    try:
        assert (mirror / "summary.json").is_file()
        for run_id in (copy, mirror.name):
            if run_id == "run-a":
                continue  # The bare name resolves to the original run, never the copy.
            with pytest.raises(ValueError, match="Not a study run"):
                record_study_episodes(root, run_id, "after_A", "A", "test", [1],
                                      root / "runs/ui-refused")
        assert resolve_study_run(root, "run-a")[0] == (root / "runs" / RUN).resolve()
        assert {row["run_id"] for row in study_catalog(root)[0]} == {RUN}
    finally:
        shutil.rmtree(root / "runs" / copy.split("/")[0])
    assert not (root / "runs/ui-refused").exists()


def test_a_repeated_bare_name_asks_for_the_full_path(lab):
    root, _ = lab
    twin = root / "runs/archive/run-a"
    shutil.copytree(root / "runs" / RUN, twin)
    try:
        with pytest.raises(ValueError, match="Several study runs are called run-a"):
            resolve_study_run(root, "run-a")
        assert resolve_study_run(root, "archive/run-a")[0] == twin.resolve()
    finally:
        shutil.rmtree(root / "runs/archive")


@pytest.mark.parametrize(("where", "value"), [("panel", "../escape"), ("key", "a/b")])
def test_unsafe_task_keys_and_panel_names_make_a_definition_invalid(tmp_path, where, value):
    study = tiny_study()
    if where == "panel":
        task_id = study["task_ids"]["A"]
        study["panels"][task_id][value] = study["panels"][task_id].pop("test")
        protocol_tasks = None
    else:
        study["task_ids"] = {value if key == "A" else key: task_id
                             for key, task_id in study["task_ids"].items()}
        protocol_tasks = {value if key == "A" else key: source for key, source in SOURCES.items()}
    with pytest.raises(ValueError, match="plain name"):
        write_definition(tmp_path, "unsafe", study,
                         **({"tasks": protocol_tasks} if protocol_tasks else {}))
