"""STUDY-001 panels: fast checks on tiny panels against the real CPU XLand."""

import json
from pathlib import Path

import numpy as np
import pytest

from ucl.envs import make_env
from ucl.study_panels import build_study, load_study, physical_identity
from ucl.tasks import load_tasks

PROTOCOL = Path(__file__).resolve().parents[1] / "configs" / "study_001" / "protocol.json"
SMALL = {"train": 4, "validation": 2, "test": 3, "final": 2, "known": 1}


@pytest.fixture(scope="module")
def study_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("study")
    protocol = json.loads(PROTOCOL.read_text())
    protocol["panels"] = SMALL
    (directory / "protocol.json").write_text(json.dumps(protocol, indent=2))
    build_study(directory / "protocol.json", directory)
    return directory


def test_identity_ignores_seed_and_detects_duplicates():
    # 5x5 navigation has few layouts, so a short seed walk must repeat a map.
    env = make_env(load_tasks()["tasks"]["development-navigation"])
    first_seen, duplicate = {}, None
    for seed in range(200):
        env.reset(seed)
        identity = physical_identity(env)
        grid = np.asarray(env._timestep.state.grid).copy()
        if identity in first_seen:
            duplicate = (first_seen[identity], (seed, grid))
            break
        first_seen[identity] = (seed, grid)
    assert duplicate is not None
    (seed_a, grid_a), (seed_b, grid_b) = duplicate
    assert seed_a != seed_b and np.array_equal(grid_a, grid_b)
    assert len(first_seen) > 1
    env.reset(seed_a)
    assert physical_identity(env) == physical_identity(env)


def test_build_produces_disjoint_panels_and_valid_manifest(study_dir):
    manifest = load_tasks(study_dir / "tasks.json")
    study = json.loads((study_dir / "panels.json").read_text())
    assert manifest["splits"]["development"] == sorted(study["task_ids"].values())
    assert study["task_ids"]["A"] == "study001-A-navigation"
    for task_id, panel in study["panels"].items():
        task = manifest["tasks"][task_id]
        assert task["height"] == task["width"] == 7
        assert task["seeds"] == panel["train"] and task["evaluation_seeds"] == panel["validation"]
        assert panel["known"] == panel["train"][:1]
        seeds = panel["train"] + panel["validation"] + panel["test"] + panel["final"]
        identities = [study["identities"][task_id][str(seed)] for seed in seeds]
        assert len(set(seeds)) == len(seeds) == 11
        assert len(set(identities)) == len(identities)
        assert study["seeds_scanned"][task_id] >= 11
    loaded = load_study(study_dir)
    assert set(loaded["tasks"]) == set(study["panels"]) and len(loaded["protocol_hash"]) == 64


def test_build_refuses_overwrite_without_force(study_dir):
    before = (study_dir / "panels.json").read_bytes()
    with pytest.raises(FileExistsError):
        build_study(study_dir / "protocol.json", study_dir)
    assert (study_dir / "panels.json").read_bytes() == before
    build_study(study_dir / "protocol.json", study_dir, force=True)
    assert (study_dir / "panels.json").read_bytes() == before


def _copy(study_dir, tmp_path):
    for name in ("protocol.json", "tasks.json", "panels.json"):
        (tmp_path / name).write_bytes((study_dir / name).read_bytes())
    return tmp_path


def test_load_study_rejects_tampered_panels(study_dir, tmp_path):
    directory = _copy(study_dir, tmp_path)
    study = json.loads((directory / "panels.json").read_text())
    task_id = study["task_ids"]["B"]
    study["panels"][task_id]["test"][0] = study["panels"][task_id]["train"][-1]
    (directory / "panels.json").write_text(json.dumps(study))
    with pytest.raises(ValueError, match="more than one panel"):
        load_study(directory)


def test_load_study_rejects_repeated_map(study_dir, tmp_path):
    directory = _copy(study_dir, tmp_path)
    study = json.loads((directory / "panels.json").read_text())
    task_id = study["task_ids"]["C"]
    panel, identities = study["panels"][task_id], study["identities"][task_id]
    identities[str(panel["final"][0])] = identities[str(panel["train"][0])]
    (directory / "panels.json").write_text(json.dumps(study))
    with pytest.raises(ValueError, match="repeats a physical map"):
        load_study(directory)


def test_load_study_rejects_edited_protocol(study_dir, tmp_path):
    directory = _copy(study_dir, tmp_path)
    protocol = json.loads((directory / "protocol.json").read_text())
    protocol["algorithm"]["lr"] = 0.01
    (directory / "protocol.json").write_text(json.dumps(protocol, indent=2))
    with pytest.raises(ValueError, match="protocol.json changed"):
        load_study(directory)


def test_the_lab_reads_a_built_study_with_the_hash_its_runs_record(study_dir):
    from ucl.study_format import panel_order, protocol_hash, read_definition

    definition = read_definition(study_dir, validate_tasks=True)
    loaded = load_study(study_dir)
    assert definition["protocol_hash"] == protocol_hash(study_dir) == loaded["protocol_hash"]
    assert definition["task_ids"] == loaded["task_ids"] and definition["sealed_panels"] == ["final"]
    assert panel_order(definition, "A") == ["validation", "test", "known", "train"]
