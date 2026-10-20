"""Event chain of preregistered study runs; training and re-evaluation are replaced by fakes."""

import json

import pytest

from ucl.cli import ROOT
from ucl.cli import main as cli_main
from ucl.study_chain import Journal, chain_status, load_plan, plan_items, plan_seeds, run_chain

PLAN = {"plan_id": "TEST-CHAIN", "study_dir": "configs/study_001",
        "architectures": ["configs/study_001/arch-LSTM.json", "configs/study_001/arch-MLP.json"],
        "tasks": ["B"], "steps": 32, "eval_every": 16, "replicas": 2, "master_seed": 123,
        "pilot_seeds": [7, 19], "seed_count": 4, "output_dir": "runs/study_001/chain-test"}


class Fakes:
    """Stand-ins that write a finished run folder and record every call in order."""

    def __init__(self, fail_run_at=None, fail_reevaluation_at=None):
        self.calls = []
        self.fail_run_at, self.fail_reevaluation_at = fail_run_at, fail_reevaluation_at

    def run(self, study, architecture, seed, tasks, *, steps, eval_every, output, root):
        self.calls.append(("run", architecture["architecture_id"], seed))
        output.mkdir(parents=True)
        if len([c for c in self.calls if c[0] == "run"]) == self.fail_run_at:
            raise RuntimeError("training crashed")
        (output / "summary.json").write_text(json.dumps({"status": "FINISHED"}))

    def reevaluate(self, root, run_id, replicas):
        self.calls.append(("reevaluate", run_id, replicas))
        if len([c for c in self.calls if c[0] == "reevaluate"]) == self.fail_reevaluation_at:
            raise RuntimeError("re-evaluation crashed")


def chain(root, fakes, **kwargs):
    return run_chain(root, PLAN, study={}, run_study_fn=fakes.run,
                     reevaluate_fn=fakes.reevaluate, **kwargs)


def test_seed_rule_keeps_pilot_seeds_and_is_reproducible():
    seeds = plan_seeds(PLAN)
    assert seeds[:2] == [7, 19] and len(seeds) == len(set(seeds)) == 4
    assert plan_seeds(PLAN) == seeds and plan_seeds({**PLAN, "master_seed": 124}) != seeds
    real = load_plan(ROOT / "plans/c08_parte1_taller.json")
    assert plan_seeds(real)[:3] == [7, 19, 31] and len(set(plan_seeds(real))) == 30
    items = plan_items(PLAN)
    assert [(i["seed"], i["name"]) for i in items[:2]] == [(7, "LSTM-s7-B"), (7, "MLP-s7-B")]
    assert len(items) == 8


def test_each_finished_step_starts_the_next_one(tmp_path):
    fakes = Fakes()
    status = chain(tmp_path, fakes)
    assert status == {"plan_id": "TEST-CHAIN", "total": 8, "finished": 8, "blocked": None}
    kinds = [call[0] for call in fakes.calls]
    assert kinds == ["run", "reevaluate"] * 8
    assert fakes.calls[1] == ("reevaluate", "study_001/chain-test/LSTM-s7-B", 2)
    events = [e["event"] for e in Journal(tmp_path, PLAN).events()]
    assert events == ["run_started", "run_finished", "reevaluation_finished"] * 8
    assert chain(tmp_path, Fakes()) == status  # nothing left: no step runs twice


def test_limit_stops_after_completed_items_and_the_next_call_continues(tmp_path):
    first = Fakes()
    assert chain(tmp_path, first, limit=3)["finished"] == 3
    second = Fakes()
    chain(tmp_path, second, limit=1)
    assert second.calls[0] == ("run", "S001-MLP", plan_seeds(PLAN)[1])


def test_a_failed_run_stops_the_chain_until_retried_with_the_same_seed(tmp_path):
    with pytest.raises(RuntimeError, match="training crashed"):
        chain(tmp_path, Fakes(fail_run_at=2))
    status = chain_status(tmp_path, PLAN)
    assert status["finished"] == 1 and status["blocked"]["status"] == "FAILED"
    assert "training crashed" in status["blocked"]["reason"]
    with pytest.raises(RuntimeError, match="--retry"):
        chain(tmp_path, Fakes())
    retried = Fakes()
    chain(tmp_path, retried, retry=True, limit=1)
    assert retried.calls[0] == ("run", "S001-MLP", 7)
    failed = tmp_path / "runs/study_001/chain-test/_failed/MLP-s7-B-1"
    assert failed.is_dir()  # the failed attempt stays as evidence
    assert chain_status(tmp_path, PLAN)["finished"] == 2


def test_a_failed_reevaluation_is_retried_without_training_again(tmp_path):
    with pytest.raises(RuntimeError, match="re-evaluation crashed"):
        chain(tmp_path, Fakes(fail_reevaluation_at=1))
    retried = Fakes()
    chain(tmp_path, retried, retry=True, limit=1)
    assert retried.calls == [("reevaluate", "study_001/chain-test/LSTM-s7-B", 2)]


def test_a_killed_process_shows_the_item_as_interrupted(tmp_path):
    journal = Journal(tmp_path, PLAN)
    journal.write("run_started", item=0, name="LSTM-s7-B", seed=7, run="x")
    blocked = chain_status(tmp_path, PLAN)["blocked"]
    assert blocked["status"] == "INTERRUPTED" and blocked["item"]["index"] == 0


def test_a_changed_plan_cannot_continue_an_old_chain(tmp_path):
    chain(tmp_path, Fakes(), limit=1)
    changed = {**PLAN, "steps": 64}
    with pytest.raises(ValueError, match="plan changed"):
        run_chain(tmp_path, changed, study={}, run_study_fn=Fakes().run,
                  reevaluate_fn=Fakes().reevaluate)


def test_plans_must_be_complete_and_write_inside_the_study_runs(tmp_path):
    incomplete = tmp_path / "a.json"
    incomplete.write_text(json.dumps({"plan_id": "X"}))
    with pytest.raises(ValueError, match="missing"):
        load_plan(incomplete)
    outside = tmp_path / "b.json"
    outside.write_text(json.dumps({**PLAN, "output_dir": "runs/elsewhere"}))
    with pytest.raises(ValueError, match="inside runs/study_001"):
        load_plan(outside)


def test_cli_status_reads_only_the_journal(tmp_path, monkeypatch, capsys):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(PLAN))
    monkeypatch.setattr("ucl.cli.ROOT", tmp_path)
    cli_main(["study-chain", "--plan", str(plan_path), "--status"])
    assert json.loads(capsys.readouterr().out) == {
        "plan_id": "TEST-CHAIN", "total": 8, "finished": 0, "blocked": None}


# ---- C01 Part II: labelled arms, unequal seed counts and a plan that waits for another -----

LR_PLAN = {"plan_id": "TEST-LR", "study_dir": "configs/study_001",
           "architectures": [
               {"config": "configs/study_001/arch-MLP.json", "label": f"MLP-lr{lr}",
                "algorithm": {"lr": lr}} for lr in (0.001, 0.003, 0.0003)],
           "tasks": ["B"], "steps": 32, "eval_every": 16, "replicas": 1, "master_seed": 9,
           "pilot_seeds": [], "seed_count": 2, "output_dir": "runs/study_001/lr-test"}
MAIN_PLAN = {"plan_id": "TEST-MAIN", "study_dir": "configs/study_001",
             "after": "plans/lr.json",
             "selection": {"rule": "highest_mean_validation_auc", "groups": ["MLP"]},
             "architectures": ["configs/study_001/arch-LSTM.json",
                               {"config": "configs/study_001/arch-MLP.json", "lr_from": "MLP",
                                "seed_count": 1}],
             "tasks": ["B"], "steps": 32, "eval_every": 16, "replicas": 2, "master_seed": 11,
             "pilot_seeds": [], "exclude_seeds": [], "seed_count": 2,
             "shuffle_within_seed": True, "output_dir": "runs/study_001/main-test"}


class ScoredFakes(Fakes):
    """Runs whose validation AUC depends on the learning rate they were given."""

    AUC = {0.001: 0.5, 0.003: 0.7, 0.0003: 0.4}

    def run(self, study, architecture, seed, tasks, *, steps, eval_every, output, root):
        lr = architecture.get("algorithm", {}).get("lr")
        self.calls.append(("run", architecture["architecture_id"], seed, lr))
        output.mkdir(parents=True)
        (output / "summary.json").write_text(json.dumps(
            {"status": "FINISHED", "auc": {"B": self.AUC.get(lr, 0.0)}}))


def write_plans(root, lr_plan=LR_PLAN):
    (root / "plans").mkdir()
    (root / "plans/lr.json").write_text(json.dumps(lr_plan))
    return MAIN_PLAN


def test_arms_keep_labels_seed_counts_exclusions_and_a_fixed_shuffle():
    items = plan_items(MAIN_PLAN)
    assert [i["label"] for i in items].count("MLP") == 1 and len(items) == 3
    assert {i["seed"] for i in items if i["label"] == "MLP"} == {plan_seeds(MAIN_PLAN)[0]}
    assert plan_items(MAIN_PLAN) == items  # the shuffle is fixed by the master seed
    seeds = plan_seeds(MAIN_PLAN)
    assert seeds[0] not in plan_seeds({**MAIN_PLAN, "exclude_seeds": [seeds[0]]})
    with pytest.raises(ValueError, match="only override"):
        plan_items({**LR_PLAN, "architectures": [
            {"config": "configs/study_001/arch-MLP.json", "algorithm": {"gamma": 0.5}}]})


def test_the_main_plan_runs_after_its_learning_rate_plan_with_the_chosen_rate(tmp_path):
    plan = write_plans(tmp_path)
    fakes = ScoredFakes()
    status = run_chain(tmp_path, plan, study={}, run_study_fn=fakes.run,
                       reevaluate_fn=fakes.reevaluate)
    runs = [call for call in fakes.calls if call[0] == "run"]
    assert len(runs) == 6 + 3 and all(call[3] is not None for call in runs[:6])
    mlp_main = [call for call in runs[6:] if call[1] == "S001-MLP"]
    assert [call[3] for call in mlp_main] == [0.003]
    assert status["finished"] == 3 and status["learning_rates"] == {"MLP": 0.003}
    resolved = Journal(tmp_path, plan).resolution()
    assert resolved["evidence"]["MLP"]["0.003"] == pytest.approx(0.7)
    again = ScoredFakes()
    run_chain(tmp_path, plan, study={}, run_study_fn=again.run, reevaluate_fn=again.reevaluate)
    assert again.calls == []  # nothing is selected or run twice


def test_a_tie_goes_to_the_learning_rate_listed_first(tmp_path):
    plan = write_plans(tmp_path)
    fakes = ScoredFakes()
    fakes.AUC = {0.001: 0.6, 0.003: 0.6, 0.0003: 0.2}
    status = run_chain(tmp_path, plan, study={}, run_study_fn=fakes.run,
                       reevaluate_fn=fakes.reevaluate)
    assert status["learning_rates"] == {"MLP": 0.001}


def test_a_failure_in_the_learning_rate_plan_blocks_the_main_plan(tmp_path):
    plan = write_plans(tmp_path)
    with pytest.raises(RuntimeError, match="training crashed"):
        run_chain(tmp_path, plan, study={}, run_study_fn=Fakes(fail_run_at=1).run,
                  reevaluate_fn=Fakes().reevaluate)
    status = chain_status(tmp_path, plan)
    assert status["after"]["blocked"]["status"] == "FAILED" and status["learning_rates"] is None
    with pytest.raises(RuntimeError, match="--retry"):
        run_chain(tmp_path, plan, study={}, run_study_fn=ScoredFakes().run,
                  reevaluate_fn=Fakes().reevaluate)
    fakes = ScoredFakes()
    status = run_chain(tmp_path, plan, retry=True, study={}, run_study_fn=fakes.run,
                       reevaluate_fn=fakes.reevaluate)
    assert status["finished"] == 3


def test_the_real_part_two_plans_are_complete():
    lr_plan = load_plan(ROOT / "plans/c01_lr_busqueda.json")
    main = load_plan(ROOT / "plans/c01_parte2_fase1.json")
    lr_seeds, main_seeds = set(plan_seeds(lr_plan)), set(plan_seeds(main))
    part_one = set(plan_seeds(load_plan(ROOT / "plans/c08_parte1_taller.json")))
    assert not (lr_seeds | part_one) & main_seeds and not part_one & lr_seeds
    items = plan_items(main)
    counts = {label: sum(i["label"] == label for i in items)
              for label in ("LSTM", "MLP-wide", "LSTM-reset", "MLP", "stack4")}
    assert counts == {"LSTM": 20, "MLP-wide": 20, "LSTM-reset": 20, "MLP": 10, "stack4": 10}
    assert len(plan_items(lr_plan)) == 18


def test_the_clock_control_reuses_the_first_ten_part_two_seeds():
    control = load_plan(ROOT / "plans/c01_parte2_sinreloj.json")
    main = load_plan(ROOT / "plans/c01_parte2_fase1.json")
    assert [i["seed"] for i in plan_items(control)] == plan_seeds(main)[:10]
    assert {i["label"] for i in plan_items(control)} == {"MLP-wide-noclock"}


def test_the_mechanism_plan_reuses_the_first_ten_part_two_seeds():
    plan = load_plan(ROOT / "plans/c01_mecanismo.json")
    main = load_plan(ROOT / "plans/c01_parte2_fase1.json")
    assert sorted({i["seed"] for i in plan_items(plan)}) == sorted(plan_seeds(main)[:10])
    assert {i["label"] for i in plan_items(plan)} == {"LSTM-reset-clock", "MLP-clock"}
