"""The sealed panel holds only new maps and is opened once, with weights fixed."""

import json
import shutil

import pytest

from ucl.cli import ROOT
from ucl.study_chain import plan_items
from ucl.study_sealed import build_sealed_panel, open_sealed, result_path


@pytest.fixture
def lab(tmp_path):
    shutil.copytree(ROOT / "configs/study_001", tmp_path / "configs/study_001")
    return tmp_path


def test_the_sealed_panel_has_only_maps_no_panel_contains(lab):
    panel = build_sealed_panel(lab, "configs/study_001", "B", 3100000, 6, "test-sealed")
    study = json.loads((lab / "configs/study_001/panels.json").read_text())
    task_id = panel["task_id"]
    old_seeds = {s for p in study["panels"][task_id].values() for s in p}
    old_maps = set(study["identities"][task_id].values())
    assert len(panel["seeds"]) == len(set(panel["identities"].values())) == 6
    assert not set(panel["seeds"]) & old_seeds
    assert not set(panel["identities"].values()) & old_maps
    with pytest.raises(FileExistsError):
        build_sealed_panel(lab, "configs/study_001", "B", 3100000, 6, "test-sealed")


PLAN = {"plan_id": "TEST-SEALED", "study_dir": "configs/study_001",
        "architectures": ["configs/study_001/arch-LSTM.json", "configs/study_001/arch-MLP.json"],
        "tasks": ["B"], "steps": 20, "eval_every": 10, "replicas": 1, "master_seed": 1,
        "pilot_seeds": [], "seed_count": 2, "output_dir": "runs/study_001/sealed-test"}


def fake_lab(lab, monkeypatch, played):
    """Runs exist only as summaries; loading and playing are replaced by recorders."""
    (lab / "plans").mkdir()
    (lab / "runs").mkdir()
    (lab / "plans/p.json").write_text(json.dumps(PLAN))
    panel = {"kind": "sealed_panel", "study_dir": "configs/study_001", "task_id":
             "study001-B-pickup", "seeds": [11, 12, 13], "identities": {}}
    from ucl.study_panels import load_study

    panel["protocol_hash"] = load_study(lab / "configs/study_001")["protocol_hash"]
    (lab / "configs/sealed").mkdir()
    (lab / "configs/sealed/p.json").write_text(json.dumps(panel))
    spec = {"analysis_id": "X", "plans": ["plans/p.json"],
            "sealed": {"evaluation_id": "TEST-OPEN", "panel": "configs/sealed/p.json",
                       "checkpoint": "best_B", "replicas": 2}}
    (lab / "spec.json").write_text(json.dumps(spec))
    summaries = {}
    for item in plan_items(PLAN):
        summaries[f"study_001/sealed-test/{item['name']}"] = {
            "status": "FINISHED", "protocol_hash": panel["protocol_hash"],
            "architecture": {}, "architecture_id": item["label"],
            "checkpoints": {"best_B": {"path": "checkpoints/best_B.pt", "sha256": "abc"}}}
    monkeypatch.setattr("ucl.study_format.resolve_study_run",
                        lambda root, run_id: (lab / "runs" / run_id, summaries[run_id]))
    monkeypatch.setattr("ucl.study_reevaluation._checked_checkpoint",
                        lambda directory, entry: directory / entry["path"])
    monkeypatch.setattr("ucl.checkpoints.load_checkpoint",
                        lambda path, expected_architecture=None: (str(path), None, None))

    def play(model, task, seed, architecture_id, sampling_seed):
        played.append((architecture_id, seed, sampling_seed))
        return {"success": seed != 13}

    monkeypatch.setattr("ucl.evaluation.evaluate_episode", play)
    return spec


def test_opening_plays_the_chosen_checkpoint_once_per_run(lab, monkeypatch):
    played = []
    fake_lab(lab, monkeypatch, played)
    status = open_sealed(lab, "spec.json")
    assert status == {"evaluation_id": "TEST-OPEN", "runs": 4, "evaluated": 4}
    assert len(played) == 4 * 3 * 2
    assert {s for _, seed, s in played if seed == 11} == {11 + 10_000_019, 11 + 2 * 10_000_019}
    run = plan_items(PLAN)[0]["name"]
    result = json.loads(result_path(lab, "TEST-OPEN", f"study_001/sealed-test/{run}").read_text())
    assert result["success_rate"] == pytest.approx(4 / 6) and result["checkpoint"] == "best_B"
    open_sealed(lab, "spec.json")
    assert len(played) == 24  # a finished run is never played again
    opening = json.loads((lab / "runs/sealed/TEST-OPEN/opening.json").read_text())
    assert opening["runs"] == 4 and opening["checkpoint"] == "best_B"


def test_the_panel_cannot_be_opened_again_with_another_spec(lab, monkeypatch):
    spec = fake_lab(lab, monkeypatch, [])
    open_sealed(lab, "spec.json")
    (lab / "spec.json").write_text(json.dumps({**spec, "note": "changed after opening"}))
    with pytest.raises(ValueError, match="already opened"):
        open_sealed(lab, "spec.json")
