"""Paired contrasts with the seed as unit, checked on synthetic data with known answers."""

import json

import numpy as np
import pytest

from ucl.cli import main as cli_main
from ucl.study_contrasts import analyze_contrasts, bca_interval, contrast_values, holm, sign_flip_p


def test_exact_sign_flip_counts_every_sign_pattern():
    assert sign_flip_p([1.0, 1.0, 1.0]) == pytest.approx(2 / 8)  # all + or all -
    assert sign_flip_p([0.3] * 10) == pytest.approx(2 / 1024)
    assert sign_flip_p([1.0, -1.0]) == 1.0
    with pytest.raises(ValueError):
        sign_flip_p([])


def test_holm_uses_the_declared_family_size():
    assert holm([0.01, 0.04], 2) == pytest.approx([0.02, 0.04])
    assert holm([0.01, 0.04], 3) == pytest.approx([0.03, 0.08])  # an untested third contrast
    assert holm([0.04, 0.01], 3) == pytest.approx([0.08, 0.03])
    with pytest.raises(ValueError, match="smaller"):
        holm([0.1, 0.2, 0.3], 2)


def test_bca_interval_covers_the_true_mean_about_95_percent_of_the_time():
    rng = np.random.default_rng(1)
    hits = 0
    for trial in range(200):
        values = rng.normal(0.1, 0.05, 20)
        low, high = bca_interval(values, resamples=2000, seed=trial)
        hits += low <= 0.1 <= high
    assert 0.88 <= hits / 200 <= 0.99


def test_contrasts_pair_arms_on_their_shared_seeds():
    scores = {"A": {1: 0.9, 2: 0.8, 3: 0.7}, "B": {1: 0.5, 3: 0.6}, "C": {1: 0.1, 2: 0.2, 3: 0.3}}
    seeds, values = contrast_values(scores, {"A": 1, "B": -1})
    assert seeds == [1, 3] and values == pytest.approx([0.4, 0.1])
    seeds, values = contrast_values(scores, {"A": 1, "C": 1, "B": -2})
    assert values == pytest.approx([0.0, -0.2])


def spec(min_effect):
    return {"bootstrap": 2000, "families": [
        {"name": "principal", "holm_m": 3, "contrasts": [
            {"id": "big", "weights": {"A": 1, "B": -1}, "min_effect": min_effect},
            {"id": "none", "weights": {"A": 1, "C": -1}, "min_effect": min_effect}]},
        {"name": "exploratory", "exploratory": True, "contrasts": [
            {"id": "e", "weights": {"B": 1, "C": -1}}]}]}


def synthetic_scores(rng, seeds=20):
    base = rng.normal(0.6, 0.05, seeds)
    return {"A": {s: b + 0.25 + rng.normal(0, 0.03) for s, b in enumerate(base)},
            "B": {s: b + rng.normal(0, 0.03) for s, b in enumerate(base)},
            "C": {s: b + 0.25 + rng.normal(0, 0.03) for s, b in enumerate(base)}}


def test_a_real_gap_is_supported_and_no_gap_is_not():
    result = analyze_contrasts(synthetic_scores(np.random.default_rng(2)), spec(0.15))
    big, none = result["families"][0]["contrasts"]
    assert big["supported"] and big["p_holm"] < 0.001
    assert big["mean"] == pytest.approx(0.25, abs=0.03)
    assert not none["supported"] and none["bca_95"][0] < 0 < none["bca_95"][1]
    explore = result["families"][1]["contrasts"][0]
    assert "supported" not in explore and "p_holm" not in explore


def test_a_gap_below_the_preregistered_effect_is_not_supported():
    result = analyze_contrasts(synthetic_scores(np.random.default_rng(3)), spec(0.30))
    big = result["families"][0]["contrasts"][0]
    assert big["p_holm"] < 0.001 and not big["supported"]


def write_runs(root, rng):
    """A tiny plan with fake finished runs and re-evaluations."""
    (root / "plans").mkdir()
    plan = {"plan_id": "TEST-CONTRASTS", "study_dir": "configs/study_001",
            "architectures": ["configs/study_001/arch-LSTM.json",
                              "configs/study_001/arch-MLP.json"],
            "tasks": ["B"], "steps": 20, "eval_every": 10, "replicas": 2, "master_seed": 3,
            "pilot_seeds": [], "seed_count": 6, "output_dir": "runs/study_001/contrasts-test"}
    (root / "plans/p.json").write_text(json.dumps(plan))
    from ucl.study_chain import plan_items

    for item in plan_items(plan):
        run_id = f"study_001/contrasts-test/{item['name']}"
        level = 0.9 if item["label"] == "LSTM" else 0.6
        (root / "runs" / run_id).mkdir(parents=True)
        (root / "runs" / run_id / "summary.json").write_text(json.dumps({
            "status": "FINISHED", "auc": {"B": level / 2},
            "curves": {"B": [{"step": 0, "success_rate": 0.1},
                             {"step": 20, "success_rate": level}]},
            "matrix": {"after_B": {"test": {"B": {"success_rate": level}}}}}))
        target = root / "runs/reevaluations" / run_id / "reevaluation.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({"status": "FINISHED", "matrix": {"after_B": {"B": {
            "replicas": [{"replica": r, "success_rate": level + rng.normal(0, 0.02)}
                         for r in (1, 2)]}}}}))
    analysis = {"analysis_id": "TEST-CONTRASTS", "plans": ["plans/p.json"],
                "scores": [{"name": "final", "kind": "final_test"},
                           {"name": "v20", "kind": "validation_at", "step": 20},
                           {"name": "auc", "kind": "auc"}],
                "bootstrap": 500, "families": [{"name": "principal", "holm_m": 1, "contrasts": [
                    {"id": "P", "weights": {"LSTM": 1, "MLP": -1}, "min_effect": 0.2}]}]}
    (root / "analysis.json").write_text(json.dumps(analysis))


def test_cli_reads_the_plans_runs_and_writes_the_analysis(tmp_path, monkeypatch, capsys):
    write_runs(tmp_path, np.random.default_rng(4))
    monkeypatch.setattr("ucl.cli.ROOT", tmp_path)
    cli_main(["study-contrasts", "--spec", str(tmp_path / "analysis.json")])
    printed = json.loads(capsys.readouterr().out)
    assert "SUPPORTED" in printed["final"]["P"] and printed["v20"]["P"].startswith("+0.300")
    written = json.loads((tmp_path / "runs/analyses/TEST-CONTRASTS/contrasts.json").read_text())
    assert written["scores"]["final"]["families"][0]["contrasts"][0]["seeds"] == 6
    assert len(written["inputs"]) == 12
