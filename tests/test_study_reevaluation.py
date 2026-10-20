"""Re-evaluation of study runs on tiny hand-built 7x7 panels; never reads the real study."""

import json
import shutil

import pytest
from test_study import ARCHITECTURE, tiny_study
from test_study_recording import write_definition

from ucl import study_reevaluation
from ucl.cli import main as cli_main
from ucl.study import run_study
from ucl.study_reevaluation import SAMPLING_OFFSET, output_path, reevaluate_run, sampling_seed

RUN = "study_001/run-r"


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    root = tmp_path_factory.mktemp("lab")
    definition = write_definition(root, "study_test", tiny_study())
    study = {key: definition[key] for key in ("protocol", "tasks", "panels", "task_ids",
                                              "protocol_hash")}
    run_study(study, ARCHITECTURE, 7, ["A"], steps=32, eval_every=16,
              output=root / "runs" / RUN, root=root)
    return root, definition


def test_sampling_seeds_are_new_for_every_replica():
    assert sampling_seed(5, 1) == 5 + SAMPLING_OFFSET
    assert sampling_seed(5, 2) == 5 + 2 * SAMPLING_OFFSET
    with pytest.raises(ValueError, match="replica 0"):
        sampling_seed(5, 0)


def test_reevaluation_replays_every_checkpoint_without_touching_the_run(lab):
    root, definition = lab
    run = root / "runs" / RUN
    before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    result = reevaluate_run(root, RUN, replicas=2)
    summary = json.loads((run / "summary.json").read_text())
    assert result["status"] == "FINISHED" and result["parameters_updated"] is False
    assert {path: path.read_bytes() for path in run.rglob("*") if path.is_file()} == before
    assert json.loads(output_path(root, RUN).read_text()) == result
    validation = definition["panels"][definition["task_ids"]["A"]]["validation"]
    curve = result["curves"]["A"]
    assert [p["step"] for p in curve] == [p["step"] for p in summary["curves"]["A"]] == [0, 16, 32]
    for point, original in zip(curve, summary["curves"]["A"]):
        assert point["checkpoint"] == original["checkpoint"] and point["seeds"] == validation
        assert point["study_success_rate"] == original["success_rate"]
        assert [r["replica"] for r in point["replicas"]] == [1, 2]
        assert all(len(r["success"]) == len(validation) for r in point["replicas"])
        assert all(r["success_rate"] == sum(r["success"]) / len(r["success"])
                   for r in point["replicas"])
    assert set(result["matrix"]) == {"initial", "after_A"}
    assert set(result["matrix"]["after_A"]) == {"A", "B", "C"}
    after = result["matrix"]["after_A"]["A"]
    assert after["sha256"] == summary["checkpoints"]["after_A"]["sha256"]
    assert after["study_success_rate"] == summary["matrix"]["after_A"]["test"]["A"]["success_rate"]


def test_a_finished_reevaluation_is_not_repeated_but_a_failed_one_can_be_retried(lab, monkeypatch):
    root, _ = lab
    run_id = "study_001/run-retry"
    shutil.copytree(root / "runs" / RUN, root / "runs" / run_id)
    original = study_reevaluation._play
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("environment crashed")
        return original(*args, **kwargs)
    monkeypatch.setattr(study_reevaluation, "_play", broken)
    with pytest.raises(RuntimeError, match="crashed"):
        reevaluate_run(root, run_id, replicas=1)
    failed = json.loads(output_path(root, run_id).read_text())
    assert failed["status"] == "FAILED" and "environment crashed" in failed["error"]
    monkeypatch.setattr(study_reevaluation, "_play", original)
    first = reevaluate_run(root, run_id, replicas=1)
    assert first["status"] == "FINISHED"
    with pytest.raises(FileExistsError, match="Already re-evaluated"):
        reevaluate_run(root, run_id, replicas=1)


def test_the_same_request_gives_the_same_outcomes(lab):
    root, _ = lab
    first = json.loads(output_path(root, RUN).read_text()) if output_path(root, RUN).is_file() \
        else reevaluate_run(root, RUN, replicas=2)
    copy_id = "study_001/run-again"
    shutil.copytree(root / "runs" / RUN, root / "runs" / copy_id)
    again = reevaluate_run(root, copy_id, replicas=2)
    assert again["curves"] == first["curves"] and again["matrix"] == first["matrix"]


def test_runs_without_per_evaluation_checkpoints_are_refused(lab):
    root, _ = lab
    old = root / "runs/study_001/run-old"
    shutil.copytree(root / "runs" / RUN, old)
    summary = json.loads((old / "summary.json").read_text())
    for point in summary["curves"]["A"]:
        point.pop("checkpoint")
    (old / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="checkpoint per evaluation"):
        reevaluate_run(root, "study_001/run-old")
    assert not output_path(root, "study_001/run-old").exists()


def test_tampered_checkpoints_and_bad_replica_counts_are_refused(lab):
    root, _ = lab
    with pytest.raises(ValueError, match="between 1 and"):
        reevaluate_run(root, RUN, replicas=0)
    bad = root / "runs/study_001/run-tampered"
    shutil.copytree(root / "runs" / RUN, bad)
    target = next((bad / "checkpoints/evaluations").glob("*.pt"))
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed after"):
        reevaluate_run(root, "study_001/run-tampered", replicas=1)
    assert json.loads(output_path(root, "study_001/run-tampered").read_text())["status"] == "FAILED"


def test_cli_reevaluates_one_run(lab, monkeypatch, capsys):
    root, _ = lab
    run_id = "study_001/run-cli"
    shutil.copytree(root / "runs" / RUN, root / "runs" / run_id)
    monkeypatch.setattr("ucl.cli.ROOT", root)
    cli_main(["study-reevaluate", "--run", run_id, "--replicas", "1"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "FINISHED" and printed["source_run"] == run_id
    assert printed["replicas"] == 1
