"""Single-environment recurrent A2C with truncated backpropagation through time.

Each rollout keeps a graph across its recurrent steps. Only rollout boundaries detach.
True terminal states have zero bootstrap; time-limit truncations bootstrap the final
observation and break return recursion before the reset. No evaluation labels enter loss.
"""
from __future__ import annotations
import copy
import hashlib
import random
import numpy as np
import torch
from torch.distributions import Categorical
from .components import detach_state


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True, warn_only=True)


def tensor_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {k: tensor_tree(v, device) for k, v in value.items()}
    raise TypeError("Recurrent states must be nested dictionaries of tensors")


def parameter_digest(model, frozen=None):
    """Bounded host memory; supports BF16 and large frozen models."""
    result = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if frozen is not None and (not parameter.requires_grad) != frozen:
            continue
        result.update(name.encode())
        result.update(str((tuple(parameter.shape), parameter.dtype)).encode())
        flattened = parameter.detach().reshape(-1)
        for start in range(0, flattened.numel(), 1_000_000):
            result.update(flattened[start:start+1_000_000].contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def discounted_returns(rewards, values_at_next, terminals, truncations, gamma):
    """Targets independent from autograd; values_at_next contains each pre-reset successor."""
    carry = values_at_next[-1].detach()
    returns = []
    for i in reversed(range(len(rewards))):
        if terminals[i]:
            carry = rewards[i]
        elif truncations[i]:
            carry = rewards[i] + gamma * values_at_next[i].detach()
        else:
            carry = rewards[i] + gamma * carry
        returns.append(carry)
    return torch.stack(returns[::-1])


class A2CTrainer:
    def __init__(self, model, env, seed, config=None, optimizer=None, seed_pool=None):
        self.model, self.env, self.seed = model, env, int(seed)
        self.config = {"lr": 0.001, "gamma": 0.99, "rollout_steps": 16,
                       "value_coef": 0.5, "entropy_coef": 0.01, "grad_clip": 0.5,
                       **(config or {})}
        if not 0 <= self.config["gamma"] <= 1 or self.config["rollout_steps"] < 1:
            raise ValueError("Invalid A2C discount or rollout length")
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("Training requires trainable parameters; use evaluate for frozen/random baselines")
        self.optimizer = optimizer or torch.optim.Adam(parameters, lr=self.config["lr"])
        self.device = next(model.parameters()).device
        self.total_steps, self.updates, self.episodes = 0, 0, 0
        self.episode_reward, self.episode_length = 0.0, 0
        self.seed_pool = self._validate_seed_pool(seed_pool)
        self.current_seed, self.action_history = self._episode_seed(0), []
        self.observation = env.reset(self.current_seed)
        self.state = model.initial_state(1, self.device)

    def _input(self, observation):
        return torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

    @staticmethod
    def _validate_seed_pool(pool):
        if pool is None:
            return None
        pool = list(pool)
        if not pool or len(set(pool)) != len(pool) or any(type(value) is not int or not 0 <= value < 2**32 for value in pool):
            raise ValueError("Training seed pool must contain unique uint32 seeds")
        return pool

    def _episode_seed(self, episode):
        if self.seed_pool is None:
            return self.seed + episode
        return self.seed_pool[(self.seed - self.seed_pool[0] + episode) % len(self.seed_pool)]

    def _environment_identity(self):
        identity = copy.deepcopy(self.env.get_replay_identity())
        for key in ("seed", "jax_initial_key"):
            identity.pop(key, None)
        return identity

    def set_task(self, env, seed, seed_pool=None):
        """Explicit boundary: retain learned weights/optimizer, reset episodic state."""
        self.env.close()
        self.env = env
        self.seed = int(seed)
        self.seed_pool = self._validate_seed_pool(seed_pool)
        self.current_seed = self._episode_seed(0)
        self.episodes = 0
        self.observation = env.reset(self.current_seed)
        self.state = self.model.initial_state(1, self.device)
        self.action_history = []
        self.episode_reward, self.episode_length = 0.0, 0

    def update(self, steps=None):
        steps = self.config["rollout_steps"] if steps is None else int(steps)
        if steps < 1:
            raise ValueError("Update requires at least one interaction")
        self.model.train()
        self.state = detach_state(self.state)
        log_probs, values, entropies, rewards, successors = [], [], [], [], []
        terminals, truncations, completed = [], [], []
        training_seeds = []
        for _ in range(steps):
            training_seeds.append(self.current_seed)
            output = self.model(self._input(self.observation), self.state)
            distribution = Categorical(logits=output["logits"])
            action = distribution.sample()
            transition = self.env.step(int(action.item()))
            self.action_history.append(int(action.item()))
            self.total_steps += 1
            self.episode_reward += transition.reward
            self.episode_length += 1
            log_probs.append(distribution.log_prob(action).squeeze(0))
            values.append(output["value"].squeeze(0))
            entropies.append(distribution.entropy().squeeze(0))
            rewards.append(torch.tensor(transition.reward, device=self.device))
            terminals.append(transition.terminated)
            truncations.append(transition.truncated)
            with torch.no_grad():
                successor = self.model(self._input(transition.observation), detach_state(output["state"]))
            successors.append(successor["value"].squeeze(0))
            self.state = output["state"]
            self.observation = transition.observation
            if transition.terminated or transition.truncated:
                completed.append({"reward": self.episode_reward, "length": self.episode_length,
                                  "success": bool(transition.info.get("success", False))})
                self.episodes += 1
                self.current_seed = self._episode_seed(self.episodes)
                self.observation = self.env.reset(self.current_seed)
                self.state = self.model.initial_state(1, self.device)
                self.action_history = []
                self.episode_reward, self.episode_length = 0.0, 0
        targets = discounted_returns(rewards, successors, terminals, truncations, self.config["gamma"])
        value_tensor = torch.stack(values)
        advantage = targets - value_tensor
        policy_loss = -(torch.stack(log_probs) * advantage.detach()).mean()
        value_loss = advantage.square().mean()
        entropy = torch.stack(entropies).mean()
        loss = policy_loss + self.config["value_coef"] * value_loss - self.config["entropy_coef"] * entropy
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.config["grad_clip"])
        if not torch.isfinite(grad_norm):
            raise RuntimeError("Nonfinite gradient")
        self.optimizer.step()
        self.updates += 1
        self.state = detach_state(self.state)
        return {"step": self.total_steps, "update": self.updates, "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()), "gradient_norm": float(grad_norm),
                "episode_reward": float(np.mean([x["reward"] for x in completed])) if completed else None,
                "episode_success": float(np.mean([x["success"] for x in completed])) if completed else None,
                "rollout_reward_mean": float(torch.stack(rewards).mean()),
                "completed_episodes": completed, "training_seeds": training_seeds}

    def state_dict(self):
        return {"seed": self.seed, "seed_pool": self.seed_pool, "environment_identity": self._environment_identity(),
                "current_seed": self.current_seed,
                "total_steps": self.total_steps, "updates": self.updates, "episodes": self.episodes,
                "episode_reward": self.episode_reward, "episode_length": self.episode_length,
                "action_history": list(self.action_history), "observation": self.observation.tolist(),
                "recurrent_state": tensor_tree(self.state, "cpu"), "config": copy.deepcopy(self.config)}

    def restore_state(self, data):
        """Rebuild exact environment state from seed/actions; no privileged state to model."""
        if data.get("environment_identity") != self._environment_identity():
            raise ValueError("Checkpoint environment identity differs or is missing; exact resume refused")
        self.seed_pool = self._validate_seed_pool(data.get("seed_pool"))
        for key in ("seed", "current_seed", "total_steps", "updates", "episodes", "episode_reward", "episode_length"):
            setattr(self, key, data[key])
        obs = self.env.reset(self.current_seed)
        for action in data["action_history"]:
            transition = self.env.step(action)
            if transition.terminated or transition.truncated:
                raise ValueError("Checkpoint action history crosses an episode boundary")
            obs = transition.observation
        if not np.array_equal(obs, np.asarray(data["observation"], dtype=np.float32)):
            raise ValueError("Environment reconstruction differs from checkpoint observation")
        self.observation, self.action_history = obs, list(data["action_history"])
        self.state = tensor_tree(data["recurrent_state"], self.device)
