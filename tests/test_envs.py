"""Tests run against the real CPU XLand implementation, never a toy substitute."""

import copy
from collections import deque

import numpy as np
import pytest

from ucl.envs import ACTIONS, EnvironmentAdapter, Transition, make_env, register_environment
from ucl.tasks import load_tasks, ruleset_identity, task_identity, validate_manifest


@pytest.fixture(scope="module")
def manifest():
    return load_tasks()


@pytest.fixture(scope="module")
def env(manifest):
    return make_env(manifest["tasks"]["development-navigation"])


def changed(task, **kwargs):
    config = copy.deepcopy(task)
    config.update(kwargs)
    config["task_sha256"] = task_identity(config)
    config["ruleset_sha256"] = ruleset_identity(config)
    return config


def test_manifest_disjoint_rulesets_and_seeds(manifest):
    hashes, seeds = [], []
    for split in ("development", "validation", "heldout"):
        selected = [manifest["tasks"][name] for name in manifest["splits"][split]]
        hashes.append({task["ruleset_sha256"] for task in selected})
        seeds.append({s for task in selected for s in task["seeds"] + task["evaluation_seeds"]})
    assert all(
        not hashes[i] & hashes[j] and not seeds[i] & seeds[j]
        for i in range(3)
        for j in range(i + 1, 3)
    )
    assert len(manifest["tasks"]) == 19


def test_manifest_detects_tampering_and_evaluation_training_leak(manifest):
    broken = copy.deepcopy(manifest)
    broken["tasks"]["development-navigation"]["goal"]["tile"][1] = 11
    with pytest.raises(ValueError, match="checksum"):
        validate_manifest(broken)
    broken = copy.deepcopy(manifest)
    broken["curricula"]["progressive"]["stages"][-1]["train_steps"] = 10
    with pytest.raises(ValueError, match="heldout"):
        validate_manifest(broken)
    broken = copy.deepcopy(manifest)
    broken["tasks"]["development-navigation"]["evaluation_seeds"] = [1000]
    with pytest.raises(ValueError, match="overlap"):
        validate_manifest(broken)


def test_return_sequence_revisits_same_task_with_new_maps(manifest):
    stages = manifest["curricula"]["return_to_previous"]["stages"]
    assert len({stage["task_id"] for stage in stages[:4]}) == 4
    assert stages[0]["task_id"] == stages[4]["task_id"] == stages[5]["task_id"]
    assert stages[0]["evaluation_seeds"] == stages[4]["evaluation_seeds"]
    assert not set(stages[4]["evaluation_seeds"]) & set(stages[5]["evaluation_seeds"])
    assert stages[4]["train_steps"] == stages[5]["train_steps"] == 0


def test_reset_dtype_dimensions_and_cpu(env):
    obs = env.reset(0)
    assert obs.dtype == np.float32 and obs.shape == (50,)
    assert env.observation_dim == 50 and env.action_count == 6
    assert np.isfinite(obs).all() and 0 <= obs.min() <= obs.max() <= 1
    assert set(env.get_replay_identity()["jax_devices"]) == {"cpu"}
    assert env.get_replay_identity()["jax_initial_key"] == [0, 0]
    assert env.agent_observation().shape == (5, 5, 2)
    assert env.agent_observation().dtype == np.uint8


def test_observations_are_copies_and_do_not_contain_privileged_fields(env):
    obs = env.reset(21)
    raw = env.agent_observation()
    expected = raw.astype(np.float32) / np.array([12, 11], dtype=np.float32)
    np.testing.assert_array_equal(obs, expected.reshape(-1))
    raw[:] = 255
    obs[:] = -1
    assert not np.any(env.agent_observation() == 255)
    # Swap hidden goal/rules without altering visible pixels: no vector change.
    visible_before = env._vector()
    state = env._timestep.state
    env._timestep = env._timestep.replace(
        state=state.replace(
            goal_encoding=state.goal_encoding * 0, rule_encoding=state.rule_encoding * 0
        )
    )
    np.testing.assert_array_equal(visible_before, env._vector())
    transition = env.step(1)
    assert set(transition.info) == {"success", "step"}
    assert set(vars(transition)) == {"observation", "reward", "terminated", "truncated", "info"}


def test_seeded_reexecution_is_exact(manifest):
    environment = make_env(manifest["tasks"]["development-hold_transform"])
    actions = [1, 0, 0, 2, 0, 5, 1, 0]

    def trajectory():
        result = [environment.reset(173)]
        for action in actions:
            item = environment.step(action)
            result.append((item.observation, item.reward, item.terminated, item.truncated))
        return result

    left, right = trajectory(), trajectory()
    np.testing.assert_array_equal(left[0], right[0])
    for a, b in zip(left[1:], right[1:]):
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]
    first_map = environment.reset(173)
    assert any(not np.array_equal(first_map, environment.reset(seed)) for seed in range(174, 179))


def test_action_mapping_and_explicit_reset(env):
    assert ACTIONS == (
        "forward",
        "turn_clockwise",
        "turn_counterclockwise",
        "pick_up",
        "put_down",
        "toggle",
    )
    env.reset(0)
    initial = int(env._timestep.state.agent.direction)
    env.step(1)
    assert int(env._timestep.state.agent.direction) == (initial + 1) % 4
    env.step(2)
    assert int(env._timestep.state.agent.direction) == initial
    for action in range(6):
        env.reset(100)
        result = env.step(action)
        assert isinstance(result, Transition) and result.info["step"] == 1


@pytest.mark.parametrize("action", [-1, 6, 2.1, True, "0", None])
def test_invalid_actions_are_rejected_before_jax(env, action):
    env.reset(1)
    with pytest.raises(ValueError, match="action"):
        env.step(action)
    assert int(env._timestep.state.step_num) == 0


def test_time_limit_is_truncation(manifest):
    config = changed(manifest["tasks"]["development-navigation"], max_steps=1)
    environment = make_env(config)
    environment.reset(7)
    result = environment.step(1)
    assert result.truncated and not result.terminated and result.reward == 0
    with pytest.raises(RuntimeError, match="reset"):
        environment.step(1)
    environment.reset(7)
    assert environment.step(1).truncated


def test_success_is_real_environment_reward_and_termination(env):
    # Evaluation-only oracle in TEST: navigate to the actual goal to verify API.
    # This oracle is not part of the agent, training pipeline or adapter outputs.
    env.reset(37)
    grid = np.asarray(env._timestep.state.grid)
    start = tuple(np.asarray(env._timestep.state.agent.position).tolist())
    goal = tuple(np.argwhere(grid[..., 0] == 6)[0])
    queue = deque([(start, [])])
    seen = {start}
    route = None
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    while queue:
        position, path = queue.popleft()
        if position == goal:
            route = path
            break
        for direction, delta in enumerate(directions):
            nxt = (position[0] + delta[0], position[1] + delta[1])
            if nxt not in seen and grid[nxt][0] in (1, 6):
                seen.add(nxt)
                queue.append((nxt, path + [direction]))
    assert route is not None and route
    for direction in route:
        while int(env._timestep.state.agent.direction) != direction:
            env.step(1)
        result = env.step(0)
    assert result.terminated and not result.truncated and result.info["success"]
    assert result.reward == pytest.approx(1 - 0.9 * result.info["step"] / 32)


def test_render_is_real_rgb_and_not_model_observation(env):
    visible = env.reset(5)
    frame = env.render()
    assert frame.dtype == np.uint8 and frame.shape == (160, 160, 3)
    assert np.unique(frame.reshape(-1, 3), axis=0).shape[0] > 4
    np.testing.assert_array_equal(visible, env._vector())
    assert frame.size != visible.size


@pytest.mark.parametrize(
    "family", ["pickup", "hold_transform", "adjacency_transform", "changed_rule", "composition"]
)
def test_real_rule_families_reset_and_step(manifest, family):
    environment = make_env(manifest["tasks"]["development-" + family])
    assert environment.reset(11).shape == (50,)
    assert environment.step(3).observation.shape == (50,)


def test_heldout_directional_uses_supported_rule_without_training(manifest):
    environment = make_env(manifest["tasks"]["heldout-directional"])
    environment.reset(202000)
    assert int(environment._timestep.state.rule_encoding[0, 0]) == 4
    assert environment.step(4).observation.shape == (50,)


def test_future_environment_registration_requires_no_runner_edit():
    class Custom(EnvironmentAdapter):
        observation_dim = 2
        action_count = 2

        def __init__(self, config):
            self.value = config["value"]

        def reset(self, seed):
            return np.array([self.value, seed], dtype=np.float32)

        def step(self, action):
            return Transition(self.reset(0), 0.0, True, False, {"success": False})

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def get_replay_identity(self):
            return {"adapter": "custom-test"}

    register_environment("custom-test", Custom)
    custom = make_env({"adapter": "custom-test", "value": 4})
    np.testing.assert_array_equal(custom.reset(1), [4, 1])
    assert custom.step(0).terminated
    with pytest.raises(ValueError, match="duplicate"):
        register_environment("custom-test", Custom)


def test_bad_configuration_is_explicit(manifest):
    with pytest.raises(ValueError, match="Unknown environment"):
        make_env({"adapter": "missing"})
    config = copy.deepcopy(manifest["tasks"]["development-navigation"])
    config["goal"]["tile"][1] = 11
    with pytest.raises(ValueError, match="checksum"):
        make_env(config)
    config = changed(manifest["tasks"]["development-navigation"], view_size=4)
    with pytest.raises(ValueError, match="view_size"):
        make_env(config)


def test_production_rules_actually_transform_and_compose(manifest):
    # Arrange a fixture solely in evaluation/test code. Agent-facing API remains unchanged.
    environment = make_env(manifest["tasks"]["development-composition"])
    environment.reset(0)
    import jax.numpy as jnp

    state = environment._timestep.state
    grid = state.grid.at[1:4, 1:4].set(jnp.array([1, 7], dtype=jnp.uint8))
    grid = grid.at[1, 2].set(jnp.array([3, 1], dtype=jnp.uint8))
    grid = grid.at[1, 3].set(jnp.array([5, 3], dtype=jnp.uint8))
    agent = state.agent.replace(
        position=jnp.array([2, 2]),
        direction=jnp.array(0),
        pocket=jnp.array([0, 0], dtype=jnp.uint8),
    )
    environment._timestep = environment._timestep.replace(
        state=state.replace(grid=grid, agent=agent)
    )
    pickup = environment.step(3)
    np.testing.assert_array_equal(np.asarray(environment._timestep.state.agent.pocket), [4, 2])
    assert not pickup.terminated
    placement = environment.step(4)
    np.testing.assert_array_equal(np.asarray(environment._timestep.state.grid[1, 3]), [11, 4])
    assert not placement.terminated
    environment.step(0)
    environment.step(1)
    final = environment.step(3)
    assert final.terminated and final.reward > 0
