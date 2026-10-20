"""Collect the numbers of the mechanism arms (probe excluded), from the lab's own files only."""
import json
from pathlib import Path

import numpy as np

from ucl.study_chain import load_plan, plan_items
from ucl.study_sealed import result_path

rng = np.random.default_rng(0)
plan = load_plan("plans/c01_mecanismo.json")
out = {}
for item in plan_items(plan):
    rid = f"study_001/c01_mecanismo/{item['name']}"
    s = json.loads(Path("runs", rid, "summary.json").read_text())
    v = json.loads(result_path(".", "C01-MECANISMO-SELLADO", rid).read_text())["success_rate"]
    out.setdefault(item["label"], []).append((
        v, s["acquisition"]["B"]["acquired"], s["acquisition"]["B"]["step"], s["auc"]["B"],
        s["parameters"]["trainable"]))
res = {}
for k, rows in out.items():
    vals = np.array([r[0] for r in rows])
    boot = rng.choice(vals, (10000, len(vals))).mean(1)
    steps = sorted(r[2] for r in rows if r[1])
    res[k] = {
        "n": len(rows), "params": rows[0][4], "sealed": float(vals.mean()),
        "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "acquired": sum(r[1] for r in rows),
        "median_step": float(np.median(steps)) if steps else None,
        "auc": float(np.mean([r[3] for r in rows]))}
c = json.loads(Path("runs/analyses/C01-MECANISMO/contrasts.json").read_text())
c = c["scores"]["sealed_best"]
res["contrasts"] = [{k: r[k] for k in ("id", "label", "mean", "bca_95", "p", "positive", "seeds")}
                    for f in c["families"] for r in f["contrasts"]]
res["ladder_small_clock"] = {
    a: float(np.mean([c["per_seed"][a][s] for s in c["per_seed"]["MLP-clock"]]))
    for a in ("MLP", "MLP-clock", "MLP-wide", "LSTM", "LSTM-reset", "LSTM-reset-clock")}
Path("paper/data/mechanism_numbers.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
