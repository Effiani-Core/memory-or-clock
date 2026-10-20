"""Small, strict, replaceable neural components. No environment dependency."""
from __future__ import annotations

from typing import Any, Callable
import math
import torch
from torch import nn

State = dict[str, torch.Tensor]
REGISTRY: dict[str, Callable[..., "Component"]] = {}


def register_component(name: str, factory: Callable[..., "Component"]) -> None:
    if not isinstance(name, str) or not name or name in REGISTRY:
        raise ValueError(f"Component name must be new and nonempty: {name!r}")
    if not callable(factory):
        raise TypeError("Component factory must be callable")
    REGISTRY[name] = factory


def create_component(name: str, input_dim: int, **params: Any) -> "Component":
    if name not in REGISTRY:
        raise ValueError(f"Unknown component {name!r}; registered: {sorted(REGISTRY)}")
    try:
        result = REGISTRY[name](input_dim=input_dim, **params)
    except TypeError as exc:
        raise ValueError(f"Invalid parameters for {name}: {exc}") from exc
    if not isinstance(result, Component):
        raise TypeError(f"{name} factory must return Component")
    return result


def detach_state(state: Any) -> Any:
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, dict):
        return {key: detach_state(value) for key, value in state.items()}
    raise TypeError("Recurrent state must consist of dictionaries and tensors")


def reset_state(state: Any, done: torch.Tensor) -> Any:
    """Explicit episode reset. True entries reset only matching batch rows."""
    if done.ndim != 1 or done.dtype != torch.bool:
        raise ValueError("Reset mask must be bool [batch]")
    if isinstance(state, torch.Tensor):
        if state.shape[0] != done.shape[0] or state.device != done.device:
            raise ValueError("Reset mask batch/device differs from recurrent state")
        mask = done.view(done.shape[0], *([1] * (state.ndim - 1)))
        return torch.where(mask, torch.zeros_like(state), state)
    if isinstance(state, dict):
        return {key: reset_state(value, done) for key, value in state.items()}
    raise TypeError("Recurrent state must consist of dictionaries and tensors")


class Component(nn.Module):
    """Contract: float32 [batch,input_dim] -> float32 [batch,output_dim]."""
    checkpoint_version = 1
    supported_devices = ("cpu", "cuda")

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        if type(input_dim) is not int or type(output_dim) is not int or min(input_dim, output_dim) <= 0:
            raise ValueError("Input/output dimensions must be positive integers")
        self.input_dim, self.output_dim = input_dim, output_dim

    def state_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def initial_state(self, batch_size: int, device: str | torch.device) -> State:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")
        return {key: torch.zeros((batch_size, *shape), device=device, dtype=torch.float32)
                for key, shape in self.state_shapes().items()}

    def validate(self, x: torch.Tensor, state: State) -> None:
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input [batch,{self.input_dim}], got {getattr(x, 'shape', type(x))}")
        if x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError("Component input must be finite float32")
        parameter = next(self.parameters(), None)
        if parameter is not None and parameter.device != x.device:
            raise ValueError("Input device differs from component parameter device")
        if not isinstance(state, dict) or set(state) != set(self.state_shapes()):
            raise ValueError(f"Recurrent state keys must be {sorted(self.state_shapes())}; explicit reset required")
        for key, shape in self.state_shapes().items():
            value = state[key]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != (x.shape[0], *shape):
                raise ValueError(f"State {key} must have shape {(x.shape[0], *shape)}; topology migration must be explicit")
            if value.device != x.device or value.dtype != x.dtype:
                raise ValueError(f"State {key} dtype/device differs from input")
            if not torch.isfinite(value).all():
                raise ValueError(f"State {key} contains nonfinite values")

    def contract(self) -> dict[str, Any]:
        trainable = [name for name, p in self.named_parameters() if p.requires_grad]
        frozen = [name for name, p in self.named_parameters() if not p.requires_grad]
        return {
            "input": {"shape": ["batch", self.input_dim], "dtype": "float32"},
            "output": {"shape": ["batch", self.output_dim], "dtype": "float32"},
            "state": {key: ["batch", *shape] for key, shape in self.state_shapes().items()},
            "trainable_parameters": trainable, "frozen_parameters": frozen,
            "supported_devices": list(self.supported_devices),
            "learning_operations": ["gradient_update"] if trainable else [],
            "checkpoint": {"version": self.checkpoint_version, "strict_shapes": True},
            "structural_operations": [], "external_memory": None,
            "instrumentation": [],
        }


class MLP(Component):
    def __init__(self, input_dim: int, output_dim: int = 32, hidden_dim: int | None = None, frozen: bool = False):
        super().__init__(input_dim, output_dim)
        width = output_dim if hidden_dim is None else hidden_dim
        if type(width) is not int or width <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        self.net = nn.Sequential(nn.Linear(input_dim, width), nn.Tanh(), nn.Linear(width, output_dim), nn.Tanh())
        if frozen:
            self.requires_grad_(False)

    def forward(self, x: torch.Tensor, state: State):
        self.validate(x, state)
        return self.net(x), {}, {}


class GRU(Component):
    def __init__(self, input_dim: int, output_dim: int = 32):
        super().__init__(input_dim, output_dim)
        self.cell = nn.GRUCell(input_dim, output_dim)

    def state_shapes(self):
        return {"h": (self.output_dim,)}

    def forward(self, x, state):
        self.validate(x, state)
        h = self.cell(x, state["h"])
        return h, {"h": h}, {"state_norm": h.detach().norm(dim=-1).tolist()}

    def contract(self):
        result = super().contract()
        result["instrumentation"] = ["state_norm"]
        return result


class LSTM(Component):
    def __init__(self, input_dim: int, output_dim: int = 32):
        super().__init__(input_dim, output_dim)
        self.cell = nn.LSTMCell(input_dim, output_dim)

    def state_shapes(self):
        return {"h": (self.output_dim,), "c": (self.output_dim,)}

    def forward(self, x, state):
        self.validate(x, state)
        h, c = self.cell(x, (state["h"], state["c"]))
        return h, {"h": h, "c": c}, {"state_norm": h.detach().norm(dim=-1).tolist(), "cell_norm": c.detach().norm(dim=-1).tolist()}

    def contract(self):
        result = super().contract()
        result["instrumentation"] = ["state_norm", "cell_norm"]
        return result


class _Mechanism(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.input_query = nn.Linear(hidden_dim, attention_dim)
        self.cell = nn.GRUCell(input_dim, hidden_dim)
        self.comm_query = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.comm_key = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.comm_value = nn.Linear(hidden_dim, hidden_dim, bias=False)


class RIM(Component):
    """Single-head RIM-inspired cell. See docs/RIM.md for deviations."""
    def __init__(self, input_dim: int, output_dim: int = 32, modules: int = 4,
                 active_modules: int = 2, attention_dim: int = 16, communication: bool = True):
        super().__init__(input_dim, output_dim)
        if any(type(v) is not int or v <= 0 for v in (modules, active_modules, attention_dim)):
            raise ValueError("RIM module counts and attention_dim must be positive integers")
        if active_modules > modules:
            raise ValueError("active_modules cannot exceed modules")
        if type(communication) is not bool:
            raise ValueError("communication must be boolean")
        self.active_modules, self.attention_dim = active_modules, attention_dim
        self.communication = communication
        self.input_key = nn.Linear(input_dim, attention_dim, bias=False)
        self.input_value = nn.Linear(input_dim, input_dim, bias=False)
        self.mechanisms = nn.ModuleList([_Mechanism(input_dim, output_dim, attention_dim) for _ in range(modules)])

    @property
    def modules_count(self):
        return len(self.mechanisms)

    def state_shapes(self):
        return {"h": (self.modules_count, self.output_dim)}

    def forward(self, x, state):
        self.validate(x, state)
        h = state["h"]
        # Null is a true zero key/value because these projections have no bias.
        tokens = torch.stack([torch.zeros_like(x), x], dim=1)
        keys, values = self.input_key(tokens), self.input_value(tokens)
        queries = torch.stack([m.input_query(h[:, i]) for i, m in enumerate(self.mechanisms)], dim=1)
        attention = torch.softmax(torch.einsum("bmd,btd->bmt", queries, keys) / math.sqrt(self.attention_dim), dim=-1)
        selected = torch.topk(1 - attention[..., 0], self.active_modules, dim=1).indices
        active = torch.zeros(h.shape[:2], dtype=torch.bool, device=x.device).scatter_(1, selected, True)
        attended = torch.einsum("bmt,btd->bmd", attention, values)
        updated = []
        for index, mechanism in enumerate(self.mechanisms):
            rows = active[:, index]
            # Only active rows execute this recurrent cell; inactive state remains an identity path.
            old = h[:, index]
            new = old.clone()
            if rows.any():
                new[rows] = mechanism.cell(attended[rows, index], old[rows])
            updated.append(new)
        intermediate = torch.stack(updated, dim=1)
        if self.communication:
            q = torch.stack([m.comm_query(intermediate[:, i]) for i, m in enumerate(self.mechanisms)], dim=1)
            k = torch.stack([m.comm_key(intermediate[:, i]) for i, m in enumerate(self.mechanisms)], dim=1)
            v = torch.stack([m.comm_value(intermediate[:, i]) for i, m in enumerate(self.mechanisms)], dim=1)
            weights = torch.softmax(torch.einsum("bmd,bnd->bmn", q, k) / math.sqrt(self.attention_dim), dim=-1)
            communicated = intermediate + torch.einsum("bmn,bnd->bmd", weights, v)
            next_h = torch.where(active.unsqueeze(-1), communicated, h)
        else:
            next_h = intermediate
        signals = {"active_mechanisms": active.detach().tolist(), "input_attention": attention.detach().tolist(),
                   "module_count": self.modules_count, "state_norm": next_h.detach().norm(dim=-1).tolist()}
        # Explicit, documented pooling keeps downstream dimensions fixed during expansion.
        return next_h.mean(dim=1), {"h": next_h}, signals

    def contract(self):
        result = super().contract()
        result["instrumentation"] = ["active_mechanisms", "input_attention", "module_count", "state_norm"]
        result["architecture_family"] = "RIM-inspired-single-head-GRU"
        return result


class ExpandableRIM(RIM):
    def __init__(self, input_dim: int, output_dim: int = 32, modules: int = 2,
                 active_modules: int = 1, attention_dim: int = 16, communication: bool = True,
                 max_modules: int = 8, max_parameters: int = 250000):
        super().__init__(input_dim, output_dim, modules, active_modules, attention_dim, communication)
        if type(max_modules) is not int or max_modules < modules:
            raise ValueError("max_modules must be an integer >= initial modules")
        if type(max_parameters) is not int or max_parameters < self.parameter_count():
            raise ValueError("max_parameters is smaller than initial component")
        self.max_modules, self.max_parameters = max_modules, max_parameters
        self.growth_events: list[dict[str, Any]] = []

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def growth_parameter_count(self, count: int):
        if type(count) is not int or count <= 0:
            raise ValueError("Growth count must be a positive integer")
        if self.modules_count + count > self.max_modules:
            raise ValueError("Growth exceeds max_modules")
        extra = count * sum(p.numel() for p in self.mechanisms[0].parameters())
        if self.parameter_count() + extra > self.max_parameters:
            raise ValueError("Growth exceeds component max_parameters")
        return extra

    def expand(self, count: int, optimizer: torch.optim.Optimizer, state: State):
        """Append new mechanisms; preserve parameter objects and their optimizer moments."""
        extra = self.growth_parameter_count(count)
        if set(state) != {"h"} or state["h"].ndim != 3 or tuple(state["h"].shape[1:]) != (self.modules_count, self.output_dim):
            raise ValueError("Growth requires current compatible recurrent state")
        reference = self.input_key.weight
        if state["h"].dtype != reference.dtype or state["h"].device != reference.device:
            raise ValueError("Growth state dtype/device differs from component")
        known = {id(p) for group in optimizer.param_groups for p in group["params"]}
        if any(id(p) not in known for p in self.parameters() if p.requires_grad):
            raise ValueError("Optimizer does not own all existing trainable component parameters")
        if any(not p.requires_grad for p in self.parameters()):
            raise ValueError("Frozen RIM cannot expand")
        before = self.modules_count
        additions = [_Mechanism(self.input_dim, self.output_dim, self.attention_dim).to(device=reference.device, dtype=reference.dtype) for _ in range(count)]
        new_parameters = [p for module in additions for p in module.parameters()]
        # A separate group is intentional; no existing optimizer state is reconstructed or dropped.
        anchor = next(self.mechanisms[0].parameters())
        original_group = next(group for group in optimizer.param_groups if any(p is anchor for p in group["params"]))
        options = {key: value for key, value in original_group.items() if key != "params"}
        optimizer.add_param_group({**options, "params": new_parameters})
        self.mechanisms.extend(additions)
        h = state["h"]
        migrated = {"h": torch.cat([h, h.new_zeros(h.shape[0], count, self.output_dim)], dim=1)}
        event = {"before_modules": before, "after_modules": self.modules_count,
                 "added_parameters": extra, "total_component_parameters": self.parameter_count(),
                 "existing_parameters_preserved": True, "optimizer_states_preserved": True}
        self.growth_events.append(event)
        return migrated, event

    def contract(self):
        result = super().contract()
        result["structural_operations"] = ["append_mechanisms_with_explicit_state_migration"]
        result["resource_budget"] = {"max_modules": self.max_modules, "max_parameters": self.max_parameters}
        return result


class ResetLSTM(LSTM):
    """C01 recurrence control: the same LSTM cell and parameters, but its state is erased
    before every step (in training and evaluation), so it cannot remember anything."""

    def forward(self, x, state):
        self.validate(x, state)
        zero = torch.zeros_like(state["h"])
        h, c = self.cell(x, (zero, zero))
        signals = {"state_norm": h.detach().norm(dim=-1).tolist(),
                   "cell_norm": c.detach().norm(dim=-1).tolist()}
        return h, {"h": torch.zeros_like(h), "c": torch.zeros_like(c)}, signals


class Clock(Component):
    """Appends the time inside the episode, t / horizon (t = 0 at the first step), to its input.
    No parameters; the count lives in the state, so an episode reset sets it back to 0."""

    def __init__(self, input_dim: int, horizon: int = 32):
        super().__init__(input_dim, input_dim + 1)
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        self.horizon = horizon

    def state_shapes(self):
        return {"t": (1,)}

    def forward(self, x, state):
        self.validate(x, state)
        return torch.cat([x, state["t"] / self.horizon], dim=-1), {"t": state["t"] + 1}, {}


class FrameStack(Component):
    """Concatenates the last ``frames`` inputs, oldest first; before the episode has that many
    steps the missing ones are zeros. A fixed, short memory with no learned recurrence."""

    def __init__(self, input_dim: int, frames: int = 4):
        if type(frames) is not int or frames < 2:
            raise ValueError("frames must be an integer >= 2")
        super().__init__(input_dim, input_dim * frames)
        self.frames = frames

    def state_shapes(self):
        return {"history": (self.input_dim * (self.frames - 1),)}

    def forward(self, x, state):
        self.validate(x, state)
        stacked = torch.cat([state["history"], x], dim=-1)
        return stacked, {"history": stacked[:, self.input_dim:]}, {}


for _name, _factory in {"mlp": MLP, "lstm": LSTM, "gru": GRU, "rim": RIM,
                        "expandable_rim": ExpandableRIM, "lstm_reset": ResetLSTM, "clock": Clock,
                        "frame_stack": FrameStack}.items():
    register_component(_name, _factory)
