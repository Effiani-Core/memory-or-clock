"""Does a trained agent's state know what time it is? A linear probe for the episode step (C01).

For each run, its chosen checkpoint plays the validation maps with fixed weights (sampling seeds
``map seed + r * 10,000,019``) and we record, at every step t, the output of one node (for the
LSTM, its hidden state after reading the observation of step t) and the observation itself. A
ridge regression then predicts t from those features, with cross-validation folds split by map so
that no map is seen in both training and testing.

Three references keep the probe honest: the same network at its initial weights (what an
untrained state encodes), the raw observation (what can be read without any memory), and the
trained features with the labels shuffled (what the probe finds by chance).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

OUTPUT_DIR = "analyses"


def ridge_cv(features, target, groups, folds: int = 4, alpha: float = 1.0) -> dict:
    """Out-of-fold R^2 and the share of steps predicted within one step, folds by group."""
    x = np.asarray(features, float)
    y = np.asarray(target, float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < folds:
        raise ValueError("Need at least as many groups as folds")
    fold_of = {g: i % folds for i, g in enumerate(unique)}
    fold = np.array([fold_of[g] for g in groups])
    predicted = np.empty_like(y)
    for k in range(folds):
        train, test = fold != k, fold == k
        mean, scale = x[train].mean(axis=0), x[train].std(axis=0) + 1e-8
        xt = np.hstack([(x[train] - mean) / scale, np.ones((train.sum(), 1))])
        penalty = alpha * np.eye(xt.shape[1])
        penalty[-1, -1] = 0.0
        weights = np.linalg.solve(xt.T @ xt + penalty, xt.T @ y[train])
        xs = np.hstack([(x[test] - mean) / scale, np.ones((test.sum(), 1))])
        predicted[test] = xs @ weights
    residual = ((y - predicted) ** 2).sum()
    total = ((y - y.mean()) ** 2).sum()
    return {"r2": float(1 - residual / total),
            "within_one_step": float((np.abs(np.round(predicted) - y) <= 1).mean()),
            "samples": int(len(y))}


def collect(model, task, node: str, seeds, replicas: int) -> dict:
    """Features at every step of every episode: the node's output and the observation."""
    from .envs import make_env
    from .study_reevaluation import sampling_seed

    states, observations, steps, groups = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for seed in seeds:
            for replica in range(1, replicas + 1):
                env = make_env(task)
                try:
                    observation = env.reset(int(seed))
                    generator = torch.Generator().manual_seed(sampling_seed(seed, replica))
                    state = model.initial_state(1, "cpu")
                    for t in range(task["max_steps"]):
                        inputs = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
                        result = model(inputs, state)
                        hidden = _node_output(model, inputs, state, node)
                        states.append(hidden)
                        observations.append(np.asarray(observation, float))
                        steps.append(t)
                        groups.append(int(seed))
                        probabilities = torch.softmax(result["logits"], -1)
                        action = int(torch.multinomial(probabilities, 1, generator=generator))
                        state = result["state"]
                        transition = env.step(action)
                        observation = transition.observation
                        if transition.terminated or transition.truncated:
                            break
                finally:
                    env.close()
    return {"state": np.array(states), "observation": np.array(observations),
            "step": np.array(steps), "group": np.array(groups)}


def _node_output(model, inputs, state, node: str) -> np.ndarray:
    """Output of ``node`` for this step (recomputed through the graph, read-only)."""
    features = {"observation": inputs}
    for spec in model.nodes:
        name = spec["id"]
        values = [features[source] for source in spec["inputs"]]
        value = values[0] if len(values) == 1 else (
            torch.cat(values, dim=-1) if spec["merge"] == "concat"
            else torch.stack(values).sum(dim=0))
        features[name], _, _ = model.components[name](value, state[name])
        if name == node:
            return features[name][0].numpy().astype(float)
    raise ValueError(f"Node {node!r} is not in the architecture")


def probe_run(root: Path, run_id: str, node: str, checkpoint: str, replicas: int,
              seed: int = 0) -> dict:
    from .checkpoints import load_checkpoint
    from .study_format import read_definition, resolve_study_run
    from .study_recording import run_definition
    from .study_reevaluation import _checked_checkpoint

    directory, summary = resolve_study_run(root, run_id)
    definition = run_definition(root, summary)
    tasks = read_definition(definition["directory"], validate_tasks=True)["tasks"]
    key = summary["task_keys"][0]
    task_id = definition["task_ids"][key]
    task, maps = tasks[task_id], definition["panels"][task_id]["validation"]
    out = {"run": run_id, "checkpoint": checkpoint}
    for name, chosen in (("trained", checkpoint), ("initial", "initial")):
        path = _checked_checkpoint(directory, summary["checkpoints"][chosen])
        model, _, _ = load_checkpoint(path, expected_architecture=summary["architecture"])
        data = collect(model, task, node, maps, replicas)
        out[f"{name}_state"] = ridge_cv(data["state"], data["step"], data["group"])
        if name == "trained":
            out["observation"] = ridge_cv(data["observation"], data["step"], data["group"])
            shuffled = np.random.default_rng(seed).permutation(data["step"])
            out["trained_state_shuffled"] = ridge_cv(data["state"], shuffled, data["group"])
            out["mean_episode_length"] = float(len(data["step"]) / (len(maps) * replicas))
    return out


def probe_plan(root, plan_path: str, labels: list[str], node: str, checkpoint: str = "best_B",
               replicas: int = 2, analysis_id: str = "C01-SONDA-TIEMPO") -> dict:
    """Probe every run of ``labels`` in a plan; writes runs/analyses/<id>/probe.json."""
    from .study_chain import load_plan, plan_items

    root = Path(root).resolve()
    plan = load_plan(root / plan_path)
    result = {"kind": "time_probe", "analysis_id": analysis_id, "plan": plan_path, "node": node,
              "checkpoint": checkpoint, "replicas": replicas, "runs": {}}
    for item in plan_items(plan):
        if item["label"] not in labels:
            continue
        run_id = (root / plan["output_dir"] / item["name"]).relative_to(root / "runs").as_posix()
        result["runs"][run_id] = {"label": item["label"], "seed": item["seed"],
                                  **probe_run(root, run_id, node, checkpoint, replicas)}
    summary = {}
    for label in labels:
        rows = [r for r in result["runs"].values() if r["label"] == label]
        summary[label] = {key: {"r2": float(np.mean([r[key]["r2"] for r in rows])),
                                "within_one_step": float(np.mean(
                                    [r[key]["within_one_step"] for r in rows]))}
                          for key in ("trained_state", "initial_state", "observation",
                                      "trained_state_shuffled")}
        summary[label]["runs"] = len(rows)
    result["summary"] = summary
    target = root / "runs" / OUTPUT_DIR / analysis_id / "probe.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=1, allow_nan=False))
    return result
