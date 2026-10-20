"""Read-only evaluation and replay artifacts. Rendering never reaches policy input."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .envs import make_env


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_episode(model, task, seed, *, output=None, architecture_id="unknown",
                     deterministic=False, sampling_seed=None):
    """Action sampling is seeded by the map seed unless an explicit sampling_seed is given."""
    env = make_env(task)
    observation = env.reset(int(seed))
    sampling = int(seed if sampling_seed is None else sampling_seed)
    random = np.random.default_rng(sampling)
    device = next(model.parameters()).device if model is not None else torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(sampling)
    state = model.initial_state(1, device) if model is not None else {}
    was_training = model.training if model is not None else False
    if model is not None:
        model.eval()
    steps, frames, observations = [], [], [observation.tolist()]
    if output is not None:
        frames.append(env.render())
    reward, success = 0.0, False
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for _ in range(task["max_steps"]):
                if model is None:
                    action, signals = int(random.integers(env.action_count)), {}
                else:
                    before = time.perf_counter()
                    inputs = torch.as_tensor(observation, dtype=torch.float32, device=device)
                    result = model(inputs.unsqueeze(0), state)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    latency = time.perf_counter() - before
                    probabilities = torch.softmax(result["logits"], -1)
                    if deterministic:
                        action = int(probabilities.argmax(-1).item())
                    else:
                        sample = torch.multinomial(probabilities, 1, generator=generator)
                        action = int(sample.item())
                    state, signals = result["state"], plain(result["signals"])
                    signals["policy_latency_seconds"] = latency
                transition = env.step(action)
                observation = transition.observation
                reward += transition.reward
                success = success or bool(transition.info.get("success", False))
                step = {"action": action, "reward": transition.reward,
                        "terminated": transition.terminated,
                        "truncated": transition.truncated, "signals": signals}
                observer = env.observer_state()
                if observer:
                    step["observer"] = observer  # privileged replay facts, never model input
                steps.append(step)
                observations.append(observation.tolist())
                if output is not None:
                    frames.append(env.render())
                if transition.terminated or transition.truncated:
                    break
        record = {"format_version": 1, "task_id": task["id"], "task_family": task["family"],
                  "split": task["split"], "seed": int(seed), "sampling_seed": sampling,
                  "architecture_id": architecture_id,
                  "steps": steps, "reward": reward, "success": success,
                  "latency_seconds": time.perf_counter() - started,
                  "environment": env.get_replay_identity(), "observations": observations,
                  "render_scope": "privileged observer display only; never model input",
                  "deterministic_actions": deterministic, "parameters_updated": False}
        if output is not None:
            directory = Path(output)
            directory.mkdir(parents=True, exist_ok=False)
            frame_path = directory / "frames.npz"
            np.savez_compressed(frame_path, frames=np.stack(frames))
            record.update(frames_file=frame_path.name,
                          frames_sha256=hashlib.sha256(frame_path.read_bytes()).hexdigest())
            (directory / "episode.json").write_text(json.dumps(record, indent=2, allow_nan=False))
        return record
    finally:
        env.close()
        if model is not None:
            model.train(was_training)


def evaluate(model, task, seeds, **kwargs):
    seeds = list(seeds)
    records = []
    base = kwargs.pop("output", None)
    for seed in seeds:
        output = Path(base) / f"{task['id']}-{seed}" if base is not None else None
        records.append(evaluate_episode(model, task, seed, output=output, **kwargs))
    if not records:
        raise ValueError("Evaluation requires at least one seed")
    rewards = [r["reward"] for r in records]
    return {"task_id": task["id"], "task_family": task["family"], "split": task["split"],
            "seeds": list(seeds), "episodes": len(records),
            "success_rate": float(np.mean([r["success"] for r in records])),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards, ddof=1)) if len(records) > 1 else 0.0,
            "latency_seconds": sum(r["latency_seconds"] for r in records),
            "environment_steps": sum(len(r["steps"]) for r in records), "records": records}
