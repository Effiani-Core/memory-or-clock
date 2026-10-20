"""Split the variance of study metrics into seed, map and evaluation noise (C08 G and D study).

Input is a finished chain plan: every run's ``reevaluation.json`` holds per-episode successes for
each map of a fixed panel, played ``R`` times with new sampling seeds. For each metric that gives
an array ``y[seed, map, replica]``. A balanced two-way crossed ANOVA (seed x map, replicas inside
each cell) estimates four variance components:

* ``seed``: real differences between training seeds (what a comparison wants to measure);
* ``map``: some maps are harder than others for everyone;
* ``seed_x_map``: a seed finds some maps easier than other seeds do;
* ``noise``: the same weights on the same map give a different result with another sampling seed.

The D study turns them into the variance of an architecture's mean score for any number of seeds,
maps and replicas, and from it the minimum number of seeds per metric for a target precision
(half-width of the 95% interval) or for a given gap between architectures (paired, by seed).
Map variance counts only when ``maps="random"`` (claims about new maps); with the study's fixed
panel it is shared by every run and does not add uncertainty to a comparison on that panel.

Seeds are resampled (paired across architectures) for percentile bootstrap intervals. The plan's
own list of runs decides what is read; nothing scans ``runs/``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

COMPONENTS = ("seed", "map", "seed_x_map", "noise")
OUTPUT_DIR = "analyses"
HALF_WIDTHS = (0.10, 0.05, 0.025)
GAPS = (0.05, 0.10, 0.20)
DESIGN_REPLICAS = (1, 3, 10)


def variance_components(y) -> dict:
    """ANOVA estimates for ``y[seed, map, replica]``; negative estimates are reported as 0."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 3:
        raise ValueError("Expected an array y[seed, map, replica]")
    seeds, maps, replicas = y.shape
    if seeds < 2 or replicas < 2:
        raise ValueError("Need at least 2 seeds and 2 replicas to separate the components")
    grand = y.mean()
    cell = y.mean(axis=2)
    seed_mean, map_mean = cell.mean(axis=1), cell.mean(axis=0)
    ms_noise = ((y - cell[..., None]) ** 2).sum() / (seeds * maps * (replicas - 1))
    ms_seed = maps * replicas * ((seed_mean - grand) ** 2).sum() / (seeds - 1)
    raw = {"noise": ms_noise}
    if maps == 1:
        # One map (or a seed-level metric): map effects cannot be separated from anything.
        raw.update(seed=(ms_seed - ms_noise) / replicas, map=0.0, seed_x_map=0.0)
    else:
        residual = cell - seed_mean[:, None] - map_mean[None, :] + grand
        ms_cross = replicas * (residual ** 2).sum() / ((seeds - 1) * (maps - 1))
        ms_map = seeds * replicas * ((map_mean - grand) ** 2).sum() / (maps - 1)
        raw.update(seed=(ms_seed - ms_cross) / (maps * replicas),
                   map=(ms_map - ms_cross) / (seeds * replicas),
                   seed_x_map=(ms_cross - ms_noise) / replicas)
    components = {name: max(float(raw[name]), 0.0) for name in COMPONENTS}
    total = sum(components.values())
    return {"mean": float(grand), "shape": {"seeds": seeds, "maps": maps, "replicas": replicas},
            "components": components, "raw": {name: float(raw[name]) for name in COMPONENTS},
            "shares": {name: (value / total if total else 0.0)
                       for name, value in components.items()}}


def per_seed_variance(components: dict, maps: int, replicas: int) -> float:
    """Variance of one seed's score around the panel mean, over ``maps`` x ``replicas``."""
    return (components["seed"] + components["seed_x_map"] / maps
            + components["noise"] / (maps * replicas))


def mean_variance(components: dict, seeds: int, maps: int, replicas: int,
                  map_effects: str = "fixed") -> float:
    """Variance of an architecture's mean score for a design of seeds x maps x replicas."""
    if map_effects not in ("fixed", "random"):
        raise ValueError("map_effects must be 'fixed' or 'random'")
    variance = per_seed_variance(components, maps, replicas) / seeds
    return variance + (components["map"] / maps if map_effects == "random" else 0.0)


def noise_share(components: dict, maps: int, replicas: int) -> float:
    """Share of the between-seed variance of run scores that is only evaluation noise."""
    total = per_seed_variance(components, maps, replicas)
    return components["noise"] / (maps * replicas) / total if total else 0.0


def minimum_seeds(components: dict, half_width: float, maps: int, replicas: int,
                  map_effects: str = "fixed", confidence: float = 0.95) -> int | None:
    """Fewest seeds whose 95% interval for the mean is at most ``half_width``; None if no
    number of seeds is enough (the map term alone is wider)."""
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    budget = (half_width / z) ** 2
    if map_effects == "random":
        budget -= components["map"] / maps
    if budget <= 0:
        return None
    return max(2, math.ceil(per_seed_variance(components, maps, replicas) / budget))


def seeds_for_gap(difference: dict, gap: float, maps: int, replicas: int,
                  alpha: float = 0.05, power: float = 0.8) -> int:
    """Seeds for a paired two-sided test to find a true gap ``gap`` with the given power."""
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return max(2, math.ceil(z ** 2 * per_seed_variance(difference, maps, replicas) / gap ** 2))


def design_report(components: dict, maps: int, replicas: int, seeds: int) -> dict:
    """Uncertainty of this design and the minimum seeds for other designs."""
    standard_error = math.sqrt(mean_variance(components, seeds, maps, replicas))
    return {
        "current": {"seeds": seeds, "maps": maps, "replicas": replicas,
                    "standard_error": standard_error, "half_width_95": 1.96 * standard_error,
                    "noise_share_one_evaluation": noise_share(components, maps, 1),
                    "noise_share": noise_share(components, maps, replicas)},
        "minimum_seeds": [
            {"half_width": h, "replicas": r, "maps": maps,
             "fixed_panel": minimum_seeds(components, h, maps, r),
             "new_maps": minimum_seeds(components, h, maps, r, map_effects="random")}
            for h in HALF_WIDTHS for r in design_replicas(replicas)]}


def design_replicas(current: int) -> list[int]:
    """Replica counts shown in the design tables, always including the current one."""
    return sorted(set(DESIGN_REPLICAS) | {current})


def _bootstrap(arrays: dict, resamples: int, seed: int) -> dict:
    """Percentile 95% intervals of every component, resampling seeds (same draw for all)."""
    if resamples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    count = next(iter(arrays.values())).shape[0]
    draws = {name: {c: [] for c in COMPONENTS} for name in arrays}
    for _ in range(resamples):
        pick = rng.integers(0, count, count)
        for name, y in arrays.items():
            for c, value in variance_components(y[pick])["components"].items():
                draws[name][c].append(value)
    return {name: {c: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                   for c, v in values.items()} for name, values in draws.items()}


# ---- reading a finished plan -------------------------------------------------------------------

def _success(replicas: list[dict]) -> np.ndarray:
    """``[map, replica]`` successes of one panel evaluation."""
    return np.array([row["success"] for row in sorted(replicas, key=lambda r: r["replica"])],
                    dtype=float).T


def run_metrics(reevaluation: dict, summary: dict, task: str, rule: dict) -> dict:
    """Per-run metric arrays: final test success and validation AUC as ``[map, replica]``, and
    acquisition (1 if the rule fires on that replica's curve) as ``[1, replica]``."""
    from .study import acquisition_step, normalized_auc

    points = reevaluation["curves"][task]
    curve = np.stack([_success(point["replicas"]) for point in points])  # [step, map, replica]
    steps = [point["step"] for point in points]

    def auc(values):
        return normalized_auc([{"step": s, "success_rate": v} for s, v in zip(steps, values)])

    area = np.array([[auc(curve[:, m, r]) for r in range(curve.shape[2])]
                     for m in range(curve.shape[1])])
    random = summary["random_baselines"][task]["validation"]["success_rate"]
    acquired = []
    for r in range(curve.shape[2]):
        rates = curve[:, :, r].mean(axis=1)
        replica_curve = [{"step": s, "success_rate": float(v)} for s, v in zip(steps, rates)]
        step = acquisition_step(replica_curve, float(rates[0]), random, **rule)
        acquired.append(float(step is not None))
    final = _success(reevaluation["matrix"][f"after_{task}"][task]["replicas"])
    return {"final_test_success": final, "validation_auc": area,
            "acquired": np.array([acquired])}


def load_plan_arrays(root, plan: dict) -> dict:
    """``{architecture: {metric: y[seed, map, replica]}}`` in seed order, plus provenance."""
    from .study_chain import plan_items
    from .study_reevaluation import output_path as reevaluation_path

    if len(plan["tasks"]) != 1:
        raise ValueError("The variance analysis reads plans with exactly one task")
    task = plan["tasks"][0]
    root = Path(root).resolve()
    protocol = json.loads((root / plan["study_dir"] / "protocol.json").read_text())
    rule = {key: protocol["acquisition"][key] for key in ("threshold", "consecutive", "min_gain")}
    output = root / plan["output_dir"]
    per_arch, inputs, missing = {}, {}, []
    expected = {}
    for item in plan_items(plan):
        run_id = (output / item["name"]).relative_to(root / "runs").as_posix()
        path = reevaluation_path(root, run_id)
        if not path.is_file():
            missing.append(item["name"])
            continue
        text = path.read_text()
        reevaluation = json.loads(text)
        if reevaluation.get("status") != "FINISHED":
            missing.append(f"{item['name']} ({reevaluation.get('status')})")
            continue
        identity = {"protocol_hash": reevaluation["protocol_hash"],
                    "replicas": reevaluation["replicas"],
                    "test_maps": reevaluation["matrix"][f"after_{task}"][task]["seeds"],
                    "validation_maps": reevaluation["curves"][task][0]["seeds"]}
        expected = expected or identity
        if identity != expected:
            raise ValueError(f"{item['name']} was re-evaluated on another protocol, panel or "
                             "number of replicas than the rest of the plan")
        summary = json.loads((root / "runs" / run_id / "summary.json").read_text())
        arch = item["label"]
        per_arch.setdefault(arch, []).append(
            (item["seed"], run_metrics(reevaluation, summary, task, rule)))
        inputs[run_id] = hashlib.sha256(text.encode()).hexdigest()
    if missing:
        raise ValueError(f"Not every run of the plan has a finished re-evaluation: {missing}")
    arrays = {arch: {metric: np.stack([metrics[metric] for _, metrics in runs])
                     for metric in runs[0][1]}
              for arch, runs in per_arch.items()}
    seeds = {arch: [seed for seed, _ in runs] for arch, runs in per_arch.items()}
    return {"task": task, "arrays": arrays, "seeds": seeds, "inputs": inputs,
            "protocol_hash": expected["protocol_hash"]}


def analyze(arrays: dict, bootstrap: int = 1000, bootstrap_seed: int = 0,
            seeds: dict | None = None) -> dict:
    """Components and designs per metric for each architecture and for paired differences
    against the first architecture (same training seed, same maps, same sampling seeds). Arms
    with fewer seeds are paired on the seeds they share with the first one."""
    names = list(arrays)
    groups = {name: arrays[name] for name in names}
    for other in names[1:]:
        if seeds is None:
            mine = theirs = slice(None)
        else:
            shared = [s for s in seeds[names[0]] if s in set(seeds[other])]
            mine = [seeds[names[0]].index(s) for s in shared]
            theirs = [seeds[other].index(s) for s in shared]
        groups[f"{names[0]} - {other}"] = {
            metric: arrays[names[0]][metric][mine] - arrays[other][metric][theirs]
            for metric in arrays[other]}
    metrics = list(arrays[names[0]])
    report = {}
    for metric in metrics:
        per_group = {name: groups[name][metric] for name in groups}
        intervals = {name: _bootstrap({name: y}, bootstrap, bootstrap_seed)[name]
                     for name, y in per_group.items()} if bootstrap > 0 else {}
        report[metric] = {}
        for name, y in per_group.items():
            estimate = variance_components(y)
            shape = estimate["shape"]
            entry = {**estimate,
                     **design_report(estimate["components"], shape["maps"],
                                     shape["replicas"], shape["seeds"])}
            if name in intervals:
                entry["bootstrap_95"] = intervals[name]
            if " - " in name:
                entry["seeds_for_gap"] = [
                    {"gap": gap, "replicas": r,
                     "seeds": seeds_for_gap(estimate["components"], gap, shape["maps"], r)}
                    for gap in GAPS for r in design_replicas(shape["replicas"])]
            report[metric][name] = entry
    return report


def analyze_plan(root, plan: dict, bootstrap: int = 1000) -> dict:
    """Analyze a finished plan and write ``runs/analyses/<plan id>/variance.json``."""
    from .study_chain import chain_status, plan_hash

    root = Path(root).resolve()
    status = chain_status(root, plan)
    if status["finished"] != status["total"]:
        raise ValueError(f"The chain has finished {status['finished']} of {status['total']} "
                         "runs; analyze it when every run is done")
    loaded = load_plan_arrays(root, plan)
    result = {"kind": "study_variance", "plan_id": plan["plan_id"], "plan_hash": plan_hash(plan),
              "protocol_hash": loaded["protocol_hash"], "task": loaded["task"],
              "seeds": loaded["seeds"], "bootstrap_resamples": bootstrap,
              "metrics": analyze(loaded["arrays"], bootstrap, seeds=loaded["seeds"]),
              "inputs": loaded["inputs"]}
    target = root / "runs" / OUTPUT_DIR / plan["plan_id"] / "variance.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=1, allow_nan=False))
    temporary.replace(target)
    return result


def brief(result: dict) -> dict:
    """The few numbers to read first: mean, component shares, noise share, seeds for +-0.05."""
    out = {}
    for metric, groups in result["metrics"].items():
        out[metric] = {}
        for name, entry in groups.items():
            fixed = next(row for row in entry["minimum_seeds"]
                         if row["half_width"] == 0.05
                         and row["replicas"] == entry["shape"]["replicas"])
            out[metric][name] = {
                "mean": round(entry["mean"], 3),
                "shares": {c: round(v, 3) for c, v in entry["shares"].items()},
                "noise_share_one_evaluation":
                    round(entry["current"]["noise_share_one_evaluation"], 3),
                "half_width_95": round(entry["current"]["half_width_95"], 3),
                "seeds_for_half_width_0.05": fixed["fixed_panel"]}
    return out
