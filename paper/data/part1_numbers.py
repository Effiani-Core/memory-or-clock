"""Collect every number Part I of the paper reports, from the lab's own files only."""
import glob
import json
from pathlib import Path

import numpy as np

from ucl.study_chain import load_plan, plan_items

ROOT = Path(".")
plan = load_plan("plans/c08_parte1_taller.json")
variance = json.loads(Path("runs/analyses/C08-PARTE1-TALLER/variance.json").read_text())
rng = np.random.default_rng(0)

runs = {}
for item in plan_items(plan):
    run_id = f"study_001/c08_parte1/{item['name']}"
    s = json.loads((ROOT / "runs" / run_id / "summary.json").read_text())
    r = json.loads((ROOT / "runs/reevaluations" / run_id / "reevaluation.json").read_text())
    rep = [row["success_rate"] for row in r["matrix"]["after_B"]["B"]["replicas"]]
    curve = s["curves"]["B"]
    runs.setdefault(item["label"], []).append({
        "seed": item["seed"],
        "test": s["matrix"]["after_B"]["test"]["B"]["success_rate"],
        "test_initial": s["matrix"]["initial"]["test"]["B"]["success_rate"],
        "test_replicas": rep,
        "auc": s["auc"]["B"],
        "acquired": s["acquisition"]["B"]["acquired"],
        "acquisition_step": s["acquisition"]["B"]["step"],
        "best_validation": s["stages"][0]["best_validation"],
        "curve_steps": [p["step"] for p in curve],
        "curve": [p["success_rate"] for p in curve],
        "curve_replicas": [[row["success_rate"] for row in pt["replicas"]]
                           for pt in r["curves"]["B"]],
        "train_seconds": s["timing"]["training_seconds"],
        "elapsed_seconds": s["elapsed_seconds"],
        "reevaluation_seconds": r["elapsed_seconds"],
        "parameters": s["parameters"]["trainable"],
        "final_sha": s["final_trainable_sha256"],
        "random_validation": s["random_baselines"]["B"]["validation"]["success_rate"],
        "random_test": s["random_baselines"]["B"]["test"]["success_rate"],
        "episodes": {"validation": curve[0]["episodes"],
                     "test": s["matrix"]["after_B"]["test"]["B"]["episodes"]},
        "versions": s["versions"]})


def iqm(values):
    v = np.sort(values)
    n = len(v)
    return float(v[n // 4: n - n // 4].mean())


def describe(values):
    v = np.asarray(values, float)
    boot = [rng.choice(v, len(v)).mean() for _ in range(10000)]
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()),
            "max": float(v.max()), "median": float(np.median(v)), "iqm": iqm(v),
            "ci95_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


out = {"arms": {}}
for arm, rows in runs.items():
    out["arms"][arm] = {
        "n": len(rows), "parameters": rows[0]["parameters"],
        "test": describe([r["test"] for r in rows]),
        "test_reevaluated_mean": describe([np.mean(r["test_replicas"]) for r in rows]),
        "auc": describe([r["auc"] for r in rows]),
        "acquired": sum(r["acquired"] for r in rows),
        "acquisition_steps": sorted(r["acquisition_step"] for r in rows if r["acquired"]),
        "test_initial_mean": float(np.mean([r["test_initial"] for r in rows])),
        "minutes_per_run": float(np.mean([r["elapsed_seconds"] for r in rows]) / 60),
        "reevaluation_minutes": float(np.mean([r["reevaluation_seconds"] for r in rows]) / 60)}
    reps = np.array([r["test_replicas"] for r in rows])
    out["arms"][arm]["replica_correlation"] = float(np.mean(
        [np.corrcoef(reps[:, a], reps[:, b])[0, 1] for a, b in ((0, 1), (0, 2), (1, 2))]))
    # How often the yes/no acquisition verdict changes between the 3 re-evaluations
    acq = variance["metrics"]["acquired"][arm]
    out["arms"][arm]["acquired_reevaluation_mean"] = acq["mean"]

L = {r["seed"]: r for r in runs["LSTM"]}
M = {r["seed"]: r for r in runs["MLP"]}
gap = np.array([L[s]["test"] - M[s]["test"] for s in L])
flips = rng.choice([-1, 1], size=(200000, len(gap)))
perm = (np.abs((flips * gap).mean(axis=1)) >= abs(gap.mean())).mean()
out["paired_gap"] = {**describe(gap), "lstm_wins": int((gap > 0).sum()), "n": len(gap),
                     "sign_flip_p_monte_carlo_200k": float(perm),
                     "auc_gap": describe([L[s]["auc"] - M[s]["auc"] for s in L])}
out["random"] = {"validation": runs["LSTM"][0]["random_validation"],
                 "test": runs["LSTM"][0]["random_test"]}
out["episodes"] = runs["LSTM"][0]["episodes"]
out["versions"] = runs["LSTM"][0]["versions"]

# Acquisition verdict instability across the 3 re-evaluations (LSTM)
from ucl.study import acquisition_step  # noqa: E402

rule = {"threshold": 0.75, "consecutive": 2, "min_gain": 0.2}
unstable = 0
for r in runs["LSTM"]:
    verdicts = []
    for k in range(3):
        rates = [pt[k] for pt in r["curve_replicas"]]
        pts = [{"step": s, "success_rate": v} for s, v in zip(r["curve_steps"], rates)]
        verdicts.append(acquisition_step(pts, rates[0], r["random_validation"], **rule))
    unstable += len({v is None for v in verdicts}) > 1
out["lstm_acquisition_verdict_changes_across_replicas"] = unstable

# Pilot reproduction: same seed, same config -> same final weights?
pilot = {}
for f in glob.glob("runs/study_001/*-pilot-S001-*-B/summary.json"):
    s = json.loads(Path(f).read_text())
    if s.get("status") == "FINISHED":
        pilot[(s["architecture_id"].split("-")[1], s["init_seed"])] = s
repro = {}
for arm, table in (("LSTM", L), ("MLP", M)):
    for seed in (7, 19, 31):
        p = pilot.get((arm, seed))
        repro[f"{arm}-{seed}"] = None if p is None else {
            "same_final_weights": p["final_trainable_sha256"] == table[seed]["final_sha"],
            "pilot_test": p["matrix"]["after_B"]["test"]["B"]["success_rate"],
            "part1_test": table[seed]["test"]}
out["pilot_reproduction"] = repro

# Variance components and designs, compact
out["variance"] = {}
for metric, groups in variance["metrics"].items():
    out["variance"][metric] = {}
    for name, e in groups.items():
        rows = {(x["half_width"], x["replicas"]): x for x in e["minimum_seeds"]}
        out["variance"][metric][name] = {
            "mean": e["mean"], "components": e["components"], "shares": e["shares"],
            "bootstrap_95": e.get("bootstrap_95"), "current": e["current"],
            "min_seeds_fixed": {f"{h}|{r}": rows[(h, r)]["fixed_panel"]
                                for h, r in rows},
            "min_seeds_new_maps": {f"{h}|{r}": rows[(h, r)]["new_maps"] for h, r in rows},
            "seeds_for_gap": {f"{x['gap']}|{x['replicas']}": x["seeds"]
                              for x in e.get("seeds_for_gap", [])}}

# Curves for the figure (study's own single evaluation)
out["curves"] = {arm: {"steps": rows[0]["curve_steps"],
                       "per_seed": [r["curve"] for r in rows]} for arm, rows in runs.items()}
out["per_seed_test"] = {arm: {str(r["seed"]): r["test"] for r in rows}
                        for arm, rows in runs.items()}
print(json.dumps({k: v for k, v in out.items() if k not in ("curves", "per_seed_test", "variance")},
                 indent=1))
Path("paper/data/part1_numbers.json").write_text(json.dumps(out, indent=1))
