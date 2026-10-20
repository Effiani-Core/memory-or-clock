"""Variance components and minimum seeds, checked on synthetic data with known answers."""

import json

import numpy as np
import pytest

from ucl.cli import main as cli_main
from ucl.study_chain import Journal, plan_items
from ucl.study_variance import (
    analyze,
    analyze_plan,
    brief,
    mean_variance,
    minimum_seeds,
    noise_share,
    seeds_for_gap,
    variance_components,
)

TRUE = {"seed": 0.04, "map": 0.02, "seed_x_map": 0.01, "noise": 0.09}


def synthetic(rng, seeds, maps, replicas, parts=TRUE):
    """y = seed + map + seed x map + noise, each normal with the given variance."""
    return (rng.normal(0, parts["seed"] ** 0.5, (seeds, 1, 1))
            + rng.normal(0, parts["map"] ** 0.5, (1, maps, 1))
            + rng.normal(0, parts["seed_x_map"] ** 0.5, (seeds, maps, 1))
            + rng.normal(0, parts["noise"] ** 0.5, (seeds, maps, replicas)))


def test_known_components_are_recovered():
    estimate = variance_components(synthetic(np.random.default_rng(1), 400, 64, 4))
    for name, value in TRUE.items():
        assert estimate["components"][name] == pytest.approx(value, rel=0.3), name
    assert sum(estimate["shares"].values()) == pytest.approx(1)


def test_coin_flips_with_no_seed_effect_are_all_evaluation_noise():
    rng = np.random.default_rng(2)
    y = (rng.random((30, 64, 3)) < 0.6).astype(float)
    estimate = variance_components(y)
    assert estimate["components"]["seed"] < 0.002
    assert noise_share(estimate["components"], 64, 1) > 0.8


def test_a_seed_level_metric_uses_one_way_components():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 0.2, (300, 1, 1)) + rng.normal(0, 0.1, (300, 1, 5))
    estimate = variance_components(y)
    assert estimate["components"]["seed"] == pytest.approx(0.04, rel=0.25)
    assert estimate["components"]["noise"] == pytest.approx(0.01, rel=0.15)
    assert estimate["components"]["map"] == estimate["components"]["seed_x_map"] == 0


def test_design_formulas_and_minimum_seeds():
    parts = {"seed": 0.01, "map": 0.02, "seed_x_map": 0.064, "noise": 0.192}
    # per seed: 0.01 + 0.064/64 + 0.192/(64*3) = 0.012
    assert mean_variance(parts, 10, 64, 3) == pytest.approx(0.0012)
    assert mean_variance(parts, 10, 64, 3, "random") == pytest.approx(0.0012 + 0.02 / 64)
    n = minimum_seeds(parts, 0.05, 64, 3)
    assert 1.96 * mean_variance(parts, n, 64, 3) ** 0.5 <= 0.05
    assert 1.96 * mean_variance(parts, n - 1, 64, 3) ** 0.5 > 0.05
    assert minimum_seeds({**parts, "map": 1.0}, 0.05, 64, 3, "random") is None
    # z(0.975) + z(0.8) = 2.8016; 2.8016**2 * 0.012 / 0.1**2 = 9.42 -> 10 seeds
    assert seeds_for_gap(parts, 0.1, 64, 3) == 10
    with pytest.raises(ValueError, match="2 replicas"):
        variance_components(np.zeros((5, 4, 1)))


def test_paired_difference_removes_the_shared_seed_effect():
    rng = np.random.default_rng(4)
    shared = synthetic(rng, 30, 64, 3)
    arrays = {"LSTM": {"m": shared + 0.25 + rng.normal(0, 0.05, shared.shape)},
              "MLP": {"m": shared + rng.normal(0, 0.05, shared.shape)}}
    report = analyze(arrays, bootstrap=50)["m"]
    difference = report["LSTM - MLP"]
    assert difference["mean"] == pytest.approx(0.25, abs=0.01)
    assert difference["components"]["seed"] < 0.001
    assert report["LSTM"]["components"]["seed"] > 0.01
    low, high = report["LSTM"]["bootstrap_95"]["seed"]
    assert low <= report["LSTM"]["components"]["seed"] <= high
    assert all(row["seeds"] == 2 for row in difference["seeds_for_gap"] if row["gap"] == 0.2)


PLAN = {"plan_id": "TEST-VARIANCE", "study_dir": "configs/study_001",
        "architectures": ["configs/study_001/arch-LSTM.json", "configs/study_001/arch-MLP.json"],
        "tasks": ["B"], "steps": 20, "eval_every": 10, "replicas": 2, "master_seed": 5,
        "pilot_seeds": [7], "seed_count": 4, "output_dir": "runs/study_001/variance-test"}
TEST_MAPS, VALIDATION_MAPS = list(range(100, 108)), list(range(200, 204))


def write_plan_outputs(root, rng, skip=None, other_panel=None):
    """Fake finished chain: journal, summaries and re-evaluations; LSTM succeeds more."""
    (root / "configs/study_001").mkdir(parents=True)
    (root / "configs/study_001/protocol.json").write_text(json.dumps(
        {"acquisition": {"threshold": 0.75, "consecutive": 2, "min_gain": 0.2}}))
    journal = Journal(root, PLAN)
    for item in plan_items(PLAN):
        run_id = f"study_001/variance-test/{item['name']}"
        for event in ("run_started", "run_finished", "reevaluation_finished"):
            journal.write(event, item=item["index"], run=run_id)
        if item["index"] == skip:
            continue
        p = 0.85 if item["name"].startswith("LSTM") else 0.55
        run = root / "runs" / run_id
        run.mkdir(parents=True)
        (run / "summary.json").write_text(json.dumps(
            {"random_baselines": {"B": {"validation": {"success_rate": 0.15}}}}))

        def replicas(maps, chance):
            return [{"replica": r, "success": (rng.random(len(maps)) < chance).astype(int).tolist()}
                    for r in (1, 2)]

        test_maps = other_panel if item["index"] == 3 and other_panel else TEST_MAPS
        reevaluation = {
            "status": "FINISHED", "protocol_hash": "h", "replicas": 2,
            "curves": {"B": [{"step": s, "seeds": VALIDATION_MAPS,
                              "replicas": replicas(VALIDATION_MAPS, chance)}
                             for s, chance in ((0, 0.1), (10, p), (20, p))]},
            "matrix": {"after_B": {"B": {"seeds": test_maps,
                                         "replicas": replicas(test_maps, p)}}}}
        target = root / "runs/reevaluations" / run_id / "reevaluation.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(reevaluation))


def test_a_finished_plan_is_analyzed_and_written(tmp_path):
    write_plan_outputs(tmp_path, np.random.default_rng(5))
    result = analyze_plan(tmp_path, PLAN, bootstrap=20)
    written = json.loads((tmp_path / "runs/analyses/TEST-VARIANCE/variance.json").read_text())
    assert written["metrics"].keys() == {"final_test_success", "validation_auc", "acquired"}
    final = result["metrics"]["final_test_success"]
    assert final["LSTM"]["shape"] == {"seeds": 4, "maps": 8, "replicas": 2}
    assert final["LSTM - MLP"]["mean"] > 0.1
    assert result["metrics"]["acquired"]["LSTM"]["shape"]["maps"] == 1
    assert len(result["inputs"]) == 8 and result["seeds"]["LSTM"][0] == 7
    assert brief(result)["validation_auc"]["MLP"]["seeds_for_half_width_0.05"] >= 2


def test_missing_or_mismatched_reevaluations_are_refused(tmp_path):
    write_plan_outputs(tmp_path / "a", np.random.default_rng(6), skip=2)
    with pytest.raises(ValueError, match="finished re-evaluation"):
        analyze_plan(tmp_path / "a", PLAN)
    write_plan_outputs(tmp_path / "b", np.random.default_rng(6), other_panel=list(range(8)))
    with pytest.raises(ValueError, match="another protocol, panel"):
        analyze_plan(tmp_path / "b", PLAN)


def test_an_unfinished_chain_is_not_analyzed(tmp_path):
    (tmp_path / "configs/study_001").mkdir(parents=True)
    with pytest.raises(ValueError, match="finished 0 of 8"):
        analyze_plan(tmp_path, PLAN)


def test_cli_writes_the_analysis_and_prints_the_brief(tmp_path, monkeypatch, capsys):
    write_plan_outputs(tmp_path, np.random.default_rng(7))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(PLAN))
    monkeypatch.setattr("ucl.cli.ROOT", tmp_path)
    cli_main(["study-variance", "--plan", str(plan_path), "--bootstrap", "10"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["final_test_success"]["LSTM - MLP"]["mean"] > 0.1
    assert (tmp_path / "runs/analyses/TEST-VARIANCE/variance.json").is_file()
