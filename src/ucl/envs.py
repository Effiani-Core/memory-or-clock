"""Architecture-independent environments; JAX stays behind this NumPy boundary."""

from __future__ import annotations

import copy
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

from .tasks import ruleset_identity, task_identity

ACTIONS = ("forward", "turn_clockwise", "turn_counterclockwise", "pick_up", "put_down", "toggle")


@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class EnvironmentAdapter(ABC):
    """Agent sees reset/step observations. Render/identity are evaluator-only."""

    observation_dim: int
    action_count: int

    @abstractmethod
    def reset(self, seed: int) -> np.ndarray: ...

    @abstractmethod
    def step(self, action: int) -> Transition: ...

    @abstractmethod
    def render(self) -> np.ndarray: ...

    @abstractmethod
    def get_replay_identity(self) -> dict: ...

    def observer_state(self) -> dict:
        """Privileged facts for the human replay (e.g. the held object). Never a model input."""
        return {}

    def close(self) -> None:
        pass


_ENVIRONMENTS: dict[str, Callable[[dict], EnvironmentAdapter]] = {}


def register_environment(name: str, factory: Callable[[dict], EnvironmentAdapter]) -> None:
    if not name or name in _ENVIRONMENTS or not callable(factory):
        raise ValueError(f"Invalid or duplicate environment registration: {name!r}")
    _ENVIRONMENTS[name] = factory


def make_env(task_config: dict) -> EnvironmentAdapter:
    name = task_config.get("adapter", "xland")
    if name not in _ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment adapter {name!r}; available: {sorted(_ENVIRONMENTS)}"
        )
    return _ENVIRONMENTS[name](copy.deepcopy(task_config))


def _jax_cpu():
    # Set before importing JAX, without influencing PyTorch's CUDA device.
    configured = os.environ.get("JAX_PLATFORMS", "cpu")
    if configured != "cpu":
        raise RuntimeError(
            "XLand requires JAX_PLATFORMS=cpu; GPU JAX is not enabled in this laboratory"
        )
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax

    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("JAX was initialized on a non-CPU device before creating XLand")
    return jax


@lru_cache(maxsize=16)
def _compiled_environment(env_id, height, width, view_size, max_steps):
    jax = _jax_cpu()
    import xminigrid

    env, params = xminigrid.make(
        env_id,
        height=height,
        width=width,
        view_size=view_size,
        max_steps=max_steps,
        render_mode="rgb_array",
    )
    # Same array shapes let all curriculum rulesets reuse these executables.
    return env, params, jax.jit(env.reset), jax.jit(env.step)


def _tile(value):
    import jax.numpy as jnp

    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("A tile must be [tile_id, color_id]")
    if any(type(x) is not int for x in value) or not (0 <= value[0] < 13 and 0 <= value[1] < 12):
        raise ValueError("Tile/color IDs are outside XLand's supported ranges")
    return jnp.asarray(value, dtype=jnp.uint8)


def _ruleset(config):
    import jax.numpy as jnp
    from xminigrid.core import goals, rules
    from xminigrid.types import RuleSet

    goal_types = {
        name: getattr(goals, name)
        for name in (
            "AgentHoldGoal",
            "AgentOnTileGoal",
            "AgentNearGoal",
            "TileNearGoal",
            "TileNearUpGoal",
            "TileNearRightGoal",
            "TileNearDownGoal",
            "TileNearLeftGoal",
        )
    }
    rule_types = {
        name: getattr(rules, name)
        for name in (
            "AgentHoldRule",
            "AgentNearRule",
            "TileNearRule",
            "TileNearUpRule",
            "TileNearRightRule",
            "TileNearDownRule",
            "TileNearLeftRule",
        )
    }

    def encode(spec, available):
        name = spec.get("type")
        if name not in available:
            raise ValueError(f"Unsupported XLand rule/goal {name!r}")
        kwargs = {key: _tile(value) for key, value in spec.items() if key != "type"}
        try:
            return available[name](**kwargs).encode().astype(jnp.uint8)
        except TypeError as exc:
            raise ValueError(f"Invalid fields for {name}: {sorted(kwargs)}") from exc

    encoded_goal = encode(config["goal"], goal_types)
    if len(config["rules"]) > 4 or not 1 <= len(config["init_tiles"]) <= 8:
        raise ValueError("Small-lab rulesets support at most 4 rules and 1..8 initial tiles")
    encoded_rules = [encode(rule, rule_types) for rule in config["rules"]]
    encoded_rules.extend([rules.EmptyRule().encode()] * (4 - len(encoded_rules)))
    tiles = [_tile(tile) for tile in config["init_tiles"]]
    tiles.extend([_tile([0, 0])] * (8 - len(tiles)))
    return RuleSet(goal=encoded_goal, rules=jnp.stack(encoded_rules), init_tiles=jnp.stack(tiles))


class XLandAdapter(EnvironmentAdapter):
    """Official XLand reset/step. Only its egocentric observation crosses to models.

    The official field of view is transparent within its crop (including behind
    walls), and does not expose the carried object. This adapter preserves both.
    """

    action_count = 6
    action_names = ACTIONS

    def __init__(self, config: dict):
        self.config = copy.deepcopy(config)
        for name in ("height", "width", "view_size", "max_steps"):
            if type(config.get(name)) is not int:
                raise ValueError(f"{name} must be an integer")
        height, width, view = config["height"], config["width"], config["view_size"]
        if not (5 <= height <= 25 and 5 <= width <= 25 and 3 <= view <= 9 and view % 2 == 1):
            raise ValueError("Grid must be 5..25 and view_size odd in 3..9")
        if not 1 <= config["max_steps"] <= 4096:
            raise ValueError("max_steps must be 1..4096")
        if config.get("task_sha256") and config["task_sha256"] != task_identity(config):
            raise ValueError("Task checksum mismatch")
        if config.get("ruleset_sha256") and config["ruleset_sha256"] != ruleset_identity(config):
            raise ValueError("Ruleset checksum mismatch")
        self._jax = _jax_cpu()
        self._env, params, self._reset, self._step = _compiled_environment(
            config["env_id"], height, width, view, config["max_steps"]
        )
        self._params = params.replace(ruleset=_ruleset(config))
        self.observation_dim = view * view * 2
        self._shape = (view, view, 2)
        self._timestep = None
        self._seed = None
        self._initial_key = None
        self._done = True

    def reset(self, seed: int) -> np.ndarray:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Seed must be a uint32 integer")
        key = self._jax.random.key(seed)
        self._initial_key = np.asarray(self._jax.random.key_data(key)).tolist()
        self._seed = seed
        self._timestep = self._reset(self._params, key)
        self._done = False
        return self._vector()

    def agent_observation(self) -> np.ndarray:
        """Copy of the native visible crop. No state, goal, rules or global grid."""
        if self._timestep is None:
            raise RuntimeError("Call reset before requesting an observation")
        observation = np.array(self._timestep.observation, dtype=np.uint8, copy=True)
        if observation.shape != self._shape:
            raise RuntimeError(
                f"Unexpected observation shape: {observation.shape}; expected {self._shape}"
            )
        return observation

    def _vector(self) -> np.ndarray:
        # Explicit documented conversion, not a neural component's implicit reshape.
        visible = self.agent_observation().astype(np.float32)
        visible /= np.array([12, 11], dtype=np.float32)
        return visible.reshape(self.observation_dim).copy()

    def step(self, action: int) -> Transition:
        if self._timestep is None or self._done:
            raise RuntimeError("Episode ended or not initialized; call reset")
        if (
            isinstance(action, (bool, np.bool_))
            or not isinstance(action, (int, np.integer))
            or not 0 <= int(action) < 6
        ):
            raise ValueError("XLand action must be an integer in [0, 5]")
        self._timestep = self._step(self._params, self._timestep, action=int(action))
        reward = float(self._timestep.reward)
        done = bool(self._timestep.last())
        terminated = done and float(self._timestep.discount) == 0.0
        truncated = done and not terminated
        self._done = done
        # Only public outcomes; no coordinates, rules, goal or hidden dynamics.
        info = {"success": terminated, "step": int(self._timestep.state.step_num)}
        return Transition(self._vector(), reward, terminated, truncated, info)

    def render(self) -> np.ndarray:
        """Privileged evaluator display, NEVER a neural observation."""
        if self._timestep is None:
            raise RuntimeError("Call reset before rendering")
        return np.array(self._env.render(self._params, self._timestep), dtype=np.uint8, copy=True)

    def observer_state(self) -> dict:
        """Held object as [tile, color]; [0, 0] is an empty pocket. Observer-only."""
        if self._timestep is None:
            raise RuntimeError("Call reset before reading the observer state")
        pocket = np.asarray(self._timestep.state.agent.pocket).tolist()
        return {"pocket": [int(value) for value in pocket]}

    def get_replay_identity(self) -> dict:
        import xminigrid

        return {
            "adapter": "xland",
            "env_id": self.config["env_id"],
            "task_id": self.config.get("id"),
            "task_sha256": task_identity(self.config),
            "ruleset_sha256": ruleset_identity(self.config),
            "seed": self._seed,
            "jax_initial_key": copy.deepcopy(self._initial_key),
            "jax_prng_impl": str(self._jax.config.jax_default_prng_impl),
            "jax_version": self._jax.__version__,
            "xminigrid_version": xminigrid.__version__,
            "jax_devices": [device.platform for device in self._jax.devices()],
            "observation_encoding": "native_egocentric_tile_color_div_12_11_row_major_v1",
            "action_names": list(ACTIONS),
            "config": copy.deepcopy(self.config),
            "privileged_metadata": True,
        }


register_environment("xland", XLandAdapter)
