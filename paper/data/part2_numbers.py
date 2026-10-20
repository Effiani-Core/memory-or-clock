"""Collect every number Part II of the paper reports, from the lab's own files only."""
import json
from pathlib import Path

import numpy as np

from ucl.study_chain import Journal, load_plan, plan_items
from ucl.study_sealed import result_path

rng = np.random.default_rng(0)
plans = [load_plan("plans/c01_parte2_fase1.json"), load_plan("plans/c01_parte2_sinreloj.json")]
arms = {}
for plan in plans:
    for item in plan_items(plan):
        run_id = f"{plan['output_dir'].removeprefix('runs/')}/{item['name']}"
        s = json.loads(Path("runs", run_id, "summary.json").read_text())
        sealed = json.loads(result_path(".", "C01-PARTE2-SELLADO", run_id).read_text())
        curve = s["curves"]["B"]
        arms.setdefault(item["label"], []).append({
            "seed": item["seed"], "sealed": sealed["success_rate"],
            "test_final": s["matrix"]["after_B"]["test"]["B"]["success_rate"],
            "acquired": s["acquisition"]["B"]["acquired"],
            "acquisition_step": s["acquisition"]["B"]["step"], "auc": s["auc"]["B"],
            "best_step": s["stages"][0]["best_validation"]["step"],
            "steps": [p["step"] for p in curve], "curve": [p["success_rate"] for p in curve],
            "parameters": s["parameters"]["trainable"], "lr": s["algorithm"]["lr"],
            "minutes": s["elapsed_seconds"] / 60})


def ci(values):
    v = np.asarray(values, float)
    boot = rng.choice(v, (10000, len(v))).mean(axis=1)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


out = {"arms": {}, "curves": {}, "per_seed_sealed": {}}
for label, rows in arms.items():
    sealed = [r["sealed"] for r in rows]
    steps = sorted(r["acquisition_step"] for r in rows if r["acquired"])
    out["arms"][label] = {
        "n": len(rows), "parameters": rows[0]["parameters"], "lr": rows[0]["lr"],
        "sealed_mean": float(np.mean(sealed)), "sealed_ci": ci(sealed),
        "sealed_sd": float(np.std(sealed, ddof=1)),
        "test_final_mean": float(np.mean([r["test_final"] for r in rows])),
        "auc_mean": float(np.mean([r["auc"] for r in rows])),
        "acquired": sum(r["acquired"] for r in rows),
        "acquisition_median": float(np.median(steps)) if steps else None,
        "minutes_per_run": float(np.mean([r["minutes"] for r in rows]))}
    out["curves"][label] = {"steps": rows[0]["steps"], "per_seed": [r["curve"] for r in rows]}
    out["per_seed_sealed"][label] = {str(r["seed"]): r["sealed"] for r in rows}

# Ladder on the 10 seeds shared by every arm
shared = [s for s in out["per_seed_sealed"]["MLP"]
          if all(s in out["per_seed_sealed"][a] for a in out["per_seed_sealed"])]
out["ladder_seeds"] = len(shared)
out["ladder"] = {a: float(np.mean([out["per_seed_sealed"][a][s] for s in shared]))
                 for a in out["per_seed_sealed"]}

for name in ("C01-PARTE2-CONFIRMATORIO", "C01-PARTE2-DESARROLLO"):
    result = json.loads(Path("runs/analyses", name, "contrasts.json").read_text())
    out[name] = {score: [{k: row[k] for k in row if k != "weights"} | {"family": f["name"]}
                         for f in block["families"] for row in f["contrasts"]]
                 for score, block in result["scores"].items()}
lr_plan = load_plan("plans/c01_parte2_fase1.json")
out["lr_search"] = Journal(".", lr_plan).resolution()
out["sealed_opening"] = json.loads(Path("runs/sealed/C01-PARTE2-SELLADO/opening.json").read_text())
panel = json.loads(Path("configs/sealed/c01_parte2_B.json").read_text())
out["sealed_panel"] = {k: panel[k] for k in ("base", "scanned", "excluded_maps")}
out["sealed_panel"]["maps"] = len(panel["seeds"])
Path("paper/data/part2_numbers.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("arms", "ladder", "ladder_seeds", "sealed_panel")},
                 indent=1))
