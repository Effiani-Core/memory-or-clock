"""The time probe finds a step that is encoded, and nothing when it is not."""

import json

import numpy as np
import torch

from ucl.architecture import GraphPolicy
from ucl.cli import ROOT, read_config
from ucl.study_probe import collect, ridge_cv


def test_ridge_recovers_an_encoded_step_and_not_noise():
    rng = np.random.default_rng(0)
    steps = np.tile(np.arange(32), 40)
    groups = np.repeat(np.arange(40), 32)
    encoded = np.c_[steps + rng.normal(0, 0.3, len(steps)), rng.normal(size=(len(steps), 5))]
    noise = rng.normal(size=(len(steps), 6))
    assert ridge_cv(encoded, steps, groups)["r2"] > 0.99
    assert ridge_cv(encoded, steps, groups)["within_one_step"] > 0.95
    assert ridge_cv(noise, steps, groups)["r2"] < 0.05


def test_folds_split_by_map_so_a_map_identity_cannot_leak():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(20), 10)
    target = np.repeat(rng.normal(size=20), 10)  # a value that only depends on the map
    one_hot = np.eye(20)[groups]
    assert ridge_cv(one_hot, target, groups)["r2"] < 0.1


def test_collect_reads_one_row_per_step_from_a_real_episode():
    config = read_config(ROOT / "configs/study_001/arch-LSTM.json")["architecture"]
    torch.manual_seed(0)
    model = GraphPolicy(config, 50, 6)
    tasks = json.loads((ROOT / "configs/study_001/tasks.json").read_text())["tasks"]
    task = tasks["study001-B-pickup"]
    data = collect(model, task, "memory", task["evaluation_seeds"][:2], replicas=1)
    assert data["state"].shape[1] == 32 and data["observation"].shape[1] == 50
    assert len(data["step"]) == len(data["group"]) == len(data["state"])
    first = data["step"][data["group"] == task["evaluation_seeds"][0]]
    assert list(first) == list(range(len(first)))
