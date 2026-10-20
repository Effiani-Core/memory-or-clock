"""Preregistered contrasts between study arms, with the seed as the unit (C01 Part II).

An analysis spec (``analyses/<name>.json``) lists the plans to read, the score of each run, and
the contrasts in families. A contrast is a weighted sum of arms (``{"LSTM": 1, "MLP-wide": -1}``)
evaluated per training seed on the seeds that every arm in it shares, so arms stay paired.

For each contrast: the mean over seeds; an exact paired sign-flip permutation test (every one of
the 2^n sign patterns, two-sided); a BCa bootstrap 95% interval (10,000 resamples by default)
and the t interval as a sensitivity check. Within a family, p-values get Holm's correction with
the family size the spec declares (which may count contrasts that were not run). A contrast is
supported when its Holm p < alpha, its mean reaches ``min_effect`` and the interval's lower
bound is above 0. Families marked exploratory are reported without correction or verdict.

Scores come only from the runs the plans list (nothing scans ``runs/``). ``final_test`` is the
end-of-training model on the test panel, averaged over the study's own evaluation and every
re-evaluation replica; ``validation_at`` and ``auc`` come from the learning curve;
``sealed`` is the run's result on a sealed panel (see ``study_sealed``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUTPUT_DIR = "analyses"
SCORES = ("final_test", "validation_at", "auc", "sealed")
MAX_EXACT = 24  # 2^24 sign patterns; beyond that the test would be Monte Carlo


def sign_flip_p(values) -> float:
    """Exact two-sided p of the mean of paired values under random signs."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0 or n > MAX_EXACT:
        raise ValueError(f"Exact sign-flip test needs 1..{MAX_EXACT} paired values")
    observed = abs(values.mean()) - 1e-12
    extreme = 0
    for start in range(0, 2 ** n, 2 ** 16):
        codes = np.arange(start, min(start + 2 ** 16, 2 ** n))
        signs = 1 - 2 * ((codes[:, None] >> np.arange(n)) & 1)
        extreme += int((np.abs((signs * values).mean(axis=1)) >= observed).sum())
    return extreme / 2 ** n


def bca_interval(values, resamples: int = 10000, seed: int = 0,
                 confidence: float = 0.95) -> list[float]:
    """Bias-corrected and accelerated bootstrap interval for the mean."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    theta = values.mean()
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, n, (resamples, n))].mean(axis=1)
    if np.all(boot == boot[0]):
        return [float(theta), float(theta)]
    z0 = stats.norm.ppf(np.clip((boot < theta).mean() + 0.5 * (boot == theta).mean(),
                                1 / resamples, 1 - 1 / resamples))
    jack = (values.sum() - values) / (n - 1)
    spread = jack.mean() - jack
    denominator = 6 * (spread ** 2).sum() ** 1.5
    accel = (spread ** 3).sum() / denominator if denominator > 0 else 0.0
    alpha = (1 - confidence) / 2
    bounds = []
    for z in stats.norm.ppf([alpha, 1 - alpha]):
        adjusted = stats.norm.cdf(z0 + (z0 + z) / (1 - accel * (z0 + z)))
        bounds.append(float(np.quantile(boot, adjusted)))
    return bounds


def t_interval(values, confidence: float = 0.95) -> list[float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    half = stats.t.ppf(0.5 + confidence / 2, n - 1) * values.std(ddof=1) / np.sqrt(n)
    return [float(values.mean() - half), float(values.mean() + half)]


def holm(p_values: list[float], family_size: int) -> list[float]:
    """Holm-adjusted p-values; ``family_size`` may exceed the number of tests run."""
    if family_size < len(p_values):
        raise ValueError("The declared family is smaller than the number of contrasts")
    order = np.argsort(p_values)
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (family_size - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def contrast_values(scores: dict, weights: dict) -> tuple[list[int], np.ndarray]:
    """Per-seed value of a weighted sum of arms, on the seeds all of its arms share."""
    missing = set(weights) - set(scores)
    if missing:
        raise ValueError(f"Contrast uses arms without scores: {sorted(missing)}")
    arms = list(weights)
    shared = [seed for seed in scores[arms[0]] if all(seed in scores[a] for a in arms[1:])]
    if len(shared) < 2:
        raise ValueError(f"Contrast {weights} has fewer than 2 shared seeds")
    values = np.array([sum(w * scores[a][seed] for a, w in weights.items()) for seed in shared])
    return shared, values


def analyze_contrasts(scores: dict, spec: dict) -> dict:
    """Every family of the spec on ``scores = {arm: {seed: value}}``."""
    alpha = spec.get("alpha", 0.05)
    resamples = spec.get("bootstrap", 10000)
    families = []
    for family in spec["families"]:
        rows = []
        for contrast in family["contrasts"]:
            seeds, values = contrast_values(scores, contrast["weights"])
            rows.append({"id": contrast["id"], "label": contrast.get("label", contrast["id"]),
                         "weights": contrast["weights"], "seeds": len(seeds),
                         "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                         "positive": int((values > 0).sum()), "zero": int((values == 0).sum()),
                         "p": sign_flip_p(values),
                         "bca_95": bca_interval(values, resamples, spec.get("seed", 0)),
                         "t_95": t_interval(values),
                         "min_effect": contrast.get("min_effect")})
        exploratory = family.get("exploratory", False)
        if not exploratory:
            for row, adjusted in zip(rows, holm([r["p"] for r in rows], family["holm_m"])):
                row["p_holm"] = adjusted
                row["supported"] = bool(adjusted < alpha and row["bca_95"][0] > 0
                                        and row["mean"] >= (row["min_effect"] or 0.0))
        families.append({"name": family["name"], "exploratory": exploratory,
                         "holm_m": family.get("holm_m"), "contrasts": rows})
    return {"alpha": alpha, "families": families,
            "arms": {arm: {"seeds": len(values), "mean": float(np.mean(list(values.values()))),
                           "sd": float(np.std(list(values.values()), ddof=1))}
                     for arm, values in scores.items()}}


# ---- reading runs ------------------------------------------------------------------------------

def run_score(root: Path, run_id: str, task: str, score: dict) -> float:
    summary = json.loads((root / "runs" / run_id / "summary.json").read_text())
    if summary.get("status") != "FINISHED":
        raise ValueError(f"{run_id} did not finish")
    kind = score["kind"]
    if kind == "final_test":
        from .study_reevaluation import output_path

        reevaluation = json.loads(output_path(root, run_id).read_text())
        if reevaluation.get("status") != "FINISHED":
            raise ValueError(f"{run_id} has no finished re-evaluation")
        row = summary["matrix"][f"after_{task}"]["test"][task]
        replicas = reevaluation["matrix"][f"after_{task}"][task]["replicas"]
        rates = [row["success_rate"]] + [r["success_rate"] for r in replicas]
        return float(np.mean(rates))
    if kind == "validation_at":
        points = [p for p in summary["curves"][task] if p["step"] == score["step"]]
        if not points:
            raise ValueError(f"{run_id} has no validation point at step {score['step']}")
        return float(points[0]["success_rate"])
    if kind == "auc":
        return float(summary["auc"][task])
    if kind == "sealed":
        from .study_sealed import result_path

        ids = score.get("evaluation_ids") or [score["evaluation_id"]]
        paths = [result_path(root, evaluation_id, run_id) for evaluation_id in ids]
        found = [path for path in paths if path.is_file()]
        if len(found) != 1:
            raise ValueError(f"{run_id} needs exactly one sealed result among {ids}")
        return float(json.loads(found[0].read_text())["success_rate"])
    raise ValueError(f"Unknown score kind {kind!r}; use one of {SCORES}")


def load_scores(root, spec: dict, score: dict) -> tuple[dict, dict]:
    from .study_chain import load_plan, plan_items

    root = Path(root).resolve()
    scores, inputs = {}, {}
    for path in spec["plans"]:
        plan = load_plan(root / path)
        task = plan["tasks"][0]
        for item in plan_items(plan):
            run_id = (root / plan["output_dir"] / item["name"]).relative_to(root / "runs")
            run_id = run_id.as_posix()
            arm = scores.setdefault(item["label"], {})
            if item["seed"] in arm:
                raise ValueError(f"Arm {item['label']} has seed {item['seed']} twice")
            arm[item["seed"]] = run_score(root, run_id, task, score)
            summary = (root / "runs" / run_id / "summary.json").read_bytes()
            inputs[run_id] = hashlib.sha256(summary).hexdigest()
    return scores, inputs


def analyze_spec(root, spec_path) -> dict:
    """Run every score of a spec and write ``runs/analyses/<analysis id>/contrasts.json``."""
    root = Path(root).resolve()
    spec_path = Path(spec_path)
    spec_path = spec_path if spec_path.is_absolute() else root / spec_path
    text = spec_path.read_text()
    spec = json.loads(text)
    result = {"kind": "study_contrasts", "analysis_id": spec["analysis_id"],
              "spec_sha256": hashlib.sha256(text.encode()).hexdigest(), "scores": {}}
    for score in spec["scores"]:
        values, inputs = load_scores(root, spec, score)
        result["scores"][score["name"]] = {"score": score, "role": score.get("role"),
                                           **analyze_contrasts(values, spec),
                                           "per_seed": {a: {str(s): v for s, v in by.items()}
                                                        for a, by in values.items()}}
        result["inputs"] = inputs
    target = root / "runs" / OUTPUT_DIR / spec["analysis_id"] / "contrasts.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=1, allow_nan=False))
    return result


def brief(result: dict) -> dict:
    out = {}
    for name, block in result["scores"].items():
        out[name] = {"arms": {a: f"{v['mean']:.3f} (n={v['seeds']})"
                              for a, v in block["arms"].items()}}
        for family in block["families"]:
            for row in family["contrasts"]:
                verdict = ("exploratory" if family["exploratory"]
                           else "SUPPORTED" if row["supported"] else "not supported")
                out[name][row["id"]] = (
                    f"{row['mean']:+.3f} BCa[{row['bca_95'][0]:+.3f},{row['bca_95'][1]:+.3f}] "
                    f"p={row['p']:.2g}" + (f" holm={row['p_holm']:.2g}" if "p_holm" in row else "")
                    + f" n={row['seeds']} -> {verdict}")
    return out
