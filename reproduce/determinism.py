"""Two identical runs (LSTM, task B, seed 7, 20k interactions) must give identical weights.
Run from the repository root. MLflow points at a dead address so tracking falls back to the
isolated local store."""
import json

from ucl import runtime  # noqa: F401
from ucl.cli import ROOT, STUDY_DIR, read_config
from ucl.study import run_study
from ucl.study_panels import load_study

study = load_study(ROOT / STUDY_DIR)
out = []
for i in (1, 2):
    target = ROOT / "runs/determinism" / f"run-{i}"
    run_study(study, read_config("configs/study_001/arch-LSTM.json"), 7, ["B"], steps=20000,
              eval_every=10000, output=target, root=ROOT, tracking={"uri": "http://127.0.0.1:9"})
    s = json.loads((target / "summary.json").read_text())
    out.append({"final_sha": s["final_trainable_sha256"],
                "checkpoints": {k: v["sha256"] for k, v in s["checkpoints"].items()}})
print(json.dumps(out, indent=1))
print("identical:", out[0] == out[1])
