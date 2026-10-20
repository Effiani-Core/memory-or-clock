"""Actual XLand integration plus clearly labelled synthetic arithmetic tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from ucl.architecture import GraphPolicy
from ucl.checkpoints import save_checkpoint, load_checkpoint, restore_rng
from ucl.components import detach_state
from ucl.envs import make_env, EnvironmentAdapter, Transition
from ucl.evaluation import evaluate_episode
from ucl.growth import GrowthPolicy
from ucl.learning import A2CTrainer, discounted_returns, parameter_digest, seed_everything
from ucl.tasks import load_tasks, task_identity, ruleset_identity
from ucl.playback import load_recording, step_details


def architecture(kind="lstm", frozen=False, **params):
    return {"nodes": [
        {"id": "encoder", "type": "mlp", "inputs": ["observation"],
         "params": {"output_dim": 8, "frozen": frozen}},
        {"id": "memory", "type": kind, "inputs": ["encoder"],
         "params": {"output_dim": 8, **params}},
    ], "output": "memory"}


def seed_cpu(seed):
    seed_everything(seed)
    torch.set_num_threads(1)


@pytest.fixture(scope="module")
def real_task():
    return load_tasks()["tasks"]["development-navigation"]


def clones(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


@pytest.mark.parametrize("kind,frozen", [("lstm", False), ("gru", True), ("rim", False)])
def test_actual_xland_a2c_updates_neural_parameters(kind, frozen, real_task):
    seed_cpu(91)
    env = make_env(real_task)
    model = GraphPolicy(architecture(kind, frozen=frozen), env.observation_dim, env.action_count)
    original = clones(model)
    trainer = A2CTrainer(model, env, 1000, {"rollout_steps": 8})
    try:
        metrics = trainer.update()
        assert metrics["step"] == 8 and metrics["update"] == 1
        assert np.isfinite(metrics["loss"]) and metrics["gradient_norm"] > 0
        assert any(not torch.equal(original[name], p) for name, p in model.named_parameters()
                   if name.startswith("components.memory"))
        if frozen:
            assert all(torch.equal(original[name], p) for name, p in model.named_parameters()
                       if name.startswith("components.encoder"))
            assert all(p.grad is None for p in model.components["encoder"].parameters())
        assert env.get_replay_identity()["jax_devices"] == ["cpu"]
    finally:
        env.close()


def test_discounted_returns_terminal_and_truncation_arithmetic():
    # Synthetic arithmetic: terminal cuts bootstrap; truncation uses final-state V,
    # not rewards from the next episode. The last ongoing state also bootstraps.
    rewards = [torch.tensor(float(value)) for value in [1, 2, 3, 4]]
    values = [torch.tensor(float(value), requires_grad=True) for value in [10, 20, 30, 40]]
    returns = discounted_returns(rewards, values, [False, True, False, False],
                                 [False, False, True, False], gamma=0.5)
    assert torch.equal(returns, torch.tensor([2., 2., 18., 24.]))
    assert not returns.requires_grad
    assert all(value.grad is None for value in values)


class ArithmeticEpisode(EnvironmentAdapter):
    """Synthetic 3-step fixture solely for checking BPTT/reset math, not XLand results."""
    observation_dim = 3
    action_count = 2
    def __init__(self, truncated=False):
        self.truncated = truncated
        self.t = 0
    def reset(self, seed):
        self.t = 0
        return np.array([1, 0, 0], dtype=np.float32)
    def step(self, action):
        self.t += 1
        done = self.t == 3
        return Transition(np.array([1, self.t / 3, float(action)], dtype=np.float32),
                          float(action == 1), done and not self.truncated,
                          done and self.truncated, {"success": done and not self.truncated})
    def render(self):
        raise AssertionError("Arithmetic fixture is not a visual environment")
    def get_replay_identity(self):
        return {"fixture": "synthetic arithmetic only"}


@pytest.mark.parametrize("truncated", [False, True])
def test_a2c_recurrent_gradients_flow_inside_rollout_and_reset(truncated):
    seed_cpu(42)
    model = GraphPolicy(architecture("lstm"), 3, 2)
    trainer = A2CTrainer(model, ArithmeticEpisode(truncated), 11, {"rollout_steps": 5})
    recorded = []
    outputs = []
    def before(_module, args):
        if torch.is_grad_enabled():
            recorded.append({"requires_grad": args[1]["h"].requires_grad,
                             "h": args[1]["h"].detach().clone()})
    def after(_module, _args, result):
        if torch.is_grad_enabled():
            result[1]["h"].retain_grad()
            outputs.append(result[1]["h"])
    pre = model.components["memory"].register_forward_pre_hook(before)
    post = model.components["memory"].register_forward_hook(after)
    try:
        metrics = trainer.update()
        assert [row["requires_grad"] for row in recorded] == [False, True, True, False, True]
        assert not recorded[0]["h"].any() and not recorded[3]["h"].any()
        assert len(metrics["completed_episodes"]) == 1
        assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in outputs)
        assert model.components["memory"].cell.weight_hh.grad.abs().sum() > 0
        assert not trainer.state["memory"]["h"].requires_grad
        recorded.clear()
        trainer.update(1)
        assert not recorded[0]["requires_grad"]
    finally:
        pre.remove()
        post.remove()


def test_expanded_xland_checkpoint_resume_next_update_is_exact(tmp_path, real_task):
    seed_cpu(81)
    env = make_env(real_task)
    model = GraphPolicy(architecture("expandable_rim", modules=2, active_modules=1, max_modules=4),
                        env.observation_dim, env.action_count)
    trainer = A2CTrainer(model, env, 1000, {"rollout_steps": 4})
    restored_env = None
    try:
        trainer.update()
        trainer.state, event = model.grow("memory", 1, trainer.optimizer, detach_state(trainer.state))
        growth = GrowthPolicy(kind="fixed_schedule", interval=1, cooldown=1)
        assert growth.observe(1, 0.0)
        growth.record_growth(1)
        trainer.update()
        checkpoint = save_checkpoint(tmp_path / "expanded.pt", model, trainer, growth=growth,
                                     metadata={"test_scope": "actual_xland_cpu_resume"})
        expected_numpy_sequence = np.random.random(6)
        expected_metrics = trainer.update()
        expected_parameters = clones(model)
        expected_optimizer = deepcopy(trainer.optimizer.state_dict())
        expected_actions = list(trainer.action_history)
        restored, optimizer, payload = load_checkpoint(checkpoint, device="cpu")
        assert restored.components["memory"].modules_count == 3
        assert len(optimizer.param_groups) == 2
        assert restored.growth_events == [event]
        assert GrowthPolicy.from_state_dict(payload["growth_policy"]).events == 1
        restored_env = make_env(real_task)
        resumed = A2CTrainer(restored, restored_env, payload["trainer"]["seed"],
                             payload["trainer"]["config"], optimizer=optimizer)
        resumed.restore_state(payload["trainer"])
        restore_rng(payload)
        np.testing.assert_array_equal(np.random.random(6), expected_numpy_sequence)
        actual_metrics = resumed.update()
        assert actual_metrics == expected_metrics
        assert resumed.action_history == expected_actions
        assert all(torch.equal(expected_parameters[name], p) for name, p in restored.named_parameters())
        actual_optimizer = optimizer.state_dict()
        assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
        for parameter_id, moments in expected_optimizer["state"].items():
            for key, value in moments.items():
                assert torch.equal(value, actual_optimizer["state"][parameter_id][key])
    finally:
        env.close()
        if restored_env is not None:
            restored_env.close()


def test_checkpoint_rejects_topology_version_and_parameter_corruption(tmp_path):
    seed_cpu(88)
    model = GraphPolicy(architecture("gru"), 4, 2)
    source = save_checkpoint(tmp_path / "original.pt", model)
    incompatible = architecture("lstm")
    with pytest.raises(ValueError, match="expected topology"):
        load_checkpoint(source, expected_architecture=incompatible)
    payload = torch.load(source, weights_only=True)
    payload["format_version"] = 999
    bad_version = tmp_path / "bad_version.pt"
    torch.save(payload, bad_version)
    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        load_checkpoint(bad_version)
    payload["format_version"] = 1
    payload["weights"].pop(next(iter(payload["weights"])))
    bad_parameter = tmp_path / "bad_parameter.pt"
    torch.save(payload, bad_parameter)
    with pytest.raises(ValueError, match="parameter names"):
        load_checkpoint(bad_parameter)


def test_actual_xland_evaluation_no_parameter_or_rng_changes_and_recorded_replay(tmp_path, real_task, monkeypatch):
    seed_cpu(84)
    model = GraphPolicy(architecture("gru"), real_task["view_size"] ** 2 * 2, 6)
    model.train()
    original = clones(model)
    before_rng = torch.get_rng_state().clone()
    original_digest = parameter_digest(model)
    output = tmp_path / "episode"
    record = evaluate_episode(model, real_task, 40000, output=output, architecture_id="test_gru")
    assert record["parameters_updated"] is False
    assert model.training
    assert torch.equal(before_rng, torch.get_rng_state())
    assert all(torch.equal(original[name], p) for name, p in model.named_parameters())
    assert original_digest == parameter_digest(model)
    assert 1 <= len(record["steps"]) <= real_task["max_steps"]
    assert record["environment"]["jax_devices"] == ["cpu"]
    # Playback uses persisted pixels/actions only, without an environment or policy.
    def forbidden(*args, **kwargs):
        raise AssertionError("Replay must not execute an environment or model")
    monkeypatch.setattr(model, "forward", forbidden)
    monkeypatch.setattr("ucl.evaluation.make_env", forbidden)
    saved = json.loads((output / "episode.json").read_text())
    frame_file = output / saved["frames_file"]
    assert hashlib.sha256(frame_file.read_bytes()).hexdigest() == saved["frames_sha256"]
    with np.load(frame_file, allow_pickle=False) as archive:
        frames = archive["frames"]
    assert frames.dtype == np.uint8
    assert len(frames) == len(saved["steps"]) + 1
    assert frames.ndim == 4 and frames.shape[-1] == 3
    replay = load_recording(output / "episode.json", tmp_path)
    assert not replay.frames.flags.writeable
    np.testing.assert_array_equal(replay.frames, frames)
    assert step_details(replay, 0)["transition"] == {}
    assert step_details(replay, 1)["transition"]["action"] == saved["steps"][0]["action"]
    assert step_details(replay, 10000)["finished"]



def test_training_seed_pool_cycles_and_task_switch_stays_inside_pool():
    seed_cpu(17)
    model = GraphPolicy(architecture("gru"), 3, 2)
    trainer = A2CTrainer(model, ArithmeticEpisode(), 14, {"rollout_steps": 10}, seed_pool=[11, 19])
    row = trainer.update()
    assert row["training_seeds"] == [19] * 3 + [11] * 3 + [19] * 3 + [11]
    assert set(row["training_seeds"]) <= {11, 19}
    state = trainer.state_dict()
    assert state["seed_pool"] == [11, 19]
    trainer.set_task(ArithmeticEpisode(), 101, seed_pool=[100, 110])
    assert trainer.current_seed == 110
    assert trainer.update(4)["training_seeds"] == [110] * 3 + [100]
    with pytest.raises(ValueError, match="unique uint32"):
        A2CTrainer(model, ArithmeticEpisode(), 0, seed_pool=[1, 1])


def test_resume_rejects_changed_hidden_rule_even_with_same_visible_observation(real_task):
    seed_cpu(18)
    original_env = make_env(real_task)
    changed_task = deepcopy(real_task)
    changed_task["rules"] = [{"type": "AgentHoldRule", "tile": [3, 1], "prod_tile": [4, 2]}]
    changed_task["task_sha256"] = task_identity(changed_task)
    changed_task["ruleset_sha256"] = ruleset_identity(changed_task)
    changed_env = make_env(changed_task)
    try:
        model = GraphPolicy(architecture("lstm"), original_env.observation_dim, original_env.action_count)
        trainer = A2CTrainer(model, original_env, 1000, seed_pool=real_task["seeds"])
        state = trainer.state_dict()
        resumed = A2CTrainer(model, changed_env, 1000, seed_pool=real_task["seeds"])
        np.testing.assert_array_equal(trainer.observation, resumed.observation)
        with pytest.raises(ValueError, match="environment identity differs"):
            resumed.restore_state(state)
    finally:
        original_env.close()
        changed_env.close()
