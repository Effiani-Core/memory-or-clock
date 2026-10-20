"""Explicit small DAG composer: representation adapters are declared nodes."""
from __future__ import annotations
from copy import deepcopy
from typing import Any
import torch
from torch import nn
from .components import Component, ExpandableRIM, RIM, create_component


class GraphPolicy(nn.Module):
    checkpoint_version = 1

    def __init__(self, config: dict[str, Any], input_dim: int, action_count: int):
        super().__init__()
        if type(action_count) is not int or action_count < 2:
            raise ValueError("action_count must be an integer >= 2")
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if not isinstance(config, dict) or not isinstance(config.get("nodes"), list) or not config["nodes"]:
            raise ValueError("Architecture must contain a nonempty nodes list")
        self._config = deepcopy(config)
        self.input_dim, self.action_count = input_dim, action_count
        self.components = nn.ModuleDict()
        self.nodes: list[dict[str, Any]] = []
        dimensions = {"observation": input_dim}
        for node in config["nodes"]:
            if not isinstance(node, dict) or not {"id", "type", "inputs"} <= set(node):
                raise ValueError("Each node requires id, type and inputs")
            name = node["id"]
            if not isinstance(name, str) or not name.isidentifier() or name in dimensions:
                raise ValueError(f"Node id must be unique identifier other than observation: {name!r}")
            inputs = node["inputs"]
            if not isinstance(inputs, list) or not inputs or any(source not in dimensions for source in inputs):
                raise ValueError(f"Node {name} must reference observation or earlier nodes; cycles/forward references are invalid")
            if len(set(inputs)) != len(inputs):
                raise ValueError(f"Node {name} repeats an input")
            merge = node.get("merge")
            if len(inputs) > 1 and merge not in ("concat", "sum"):
                raise ValueError(f"Multi-input node {name} requires explicit merge: concat or sum")
            if merge is not None and merge not in ("concat", "sum"):
                raise ValueError(f"Unknown merge method {merge!r}")
            widths = [dimensions[source] for source in inputs]
            if merge == "sum" and len(set(widths)) != 1:
                raise ValueError(f"Sum inputs for node {name} must have equal dimensions; declare an adapter")
            width = sum(widths) if merge == "concat" else widths[0]
            params = node.get("params", {})
            if not isinstance(params, dict):
                raise ValueError(f"Node {name} params must be an object")
            component = create_component(node["type"], input_dim=width, **params)
            self.components[name] = component
            dimensions[name] = component.output_dim
            self.nodes.append(deepcopy(node))
        output = config.get("output")
        if output not in self.components:
            raise ValueError("Architecture output must name a component node")
        self.output_node = output
        self.policy_head = nn.Linear(dimensions[output], action_count)
        self.value_head = nn.Linear(dimensions[output], 1)
        self.max_parameters = config.get("max_parameters", 1000000)
        if type(self.max_parameters) is not int or self.max_parameters <= 0:
            raise ValueError("Architecture max_parameters must be a positive integer")
        if self.parameter_counts()["trainable"] > self.max_parameters:
            raise ValueError("Architecture exceeds configured trainable max_parameters")
        self.growth_events: list[dict[str, Any]] = []

    @property
    def config(self):
        result = deepcopy(self._config)
        for node in result["nodes"]:
            component = self.components[node["id"]]
            if isinstance(component, RIM):
                node.setdefault("params", {})["modules"] = component.modules_count
        return result

    def parameter_counts(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}

    def initial_state(self, batch_size: int, device: str | torch.device):
        return {name: component.initial_state(batch_size, device) for name, component in self.components.items()}

    def contract(self):
        return {"input_dim": self.input_dim, "action_count": self.action_count,
                "checkpoint_version": self.checkpoint_version,
                "components": {name: component.contract() for name, component in self.components.items()},
                "parameters": self.parameter_counts(), "config": self.config}

    def forward(self, obs: torch.Tensor, state=None):
        if not isinstance(obs, torch.Tensor) or obs.ndim != 2 or obs.shape[1] != self.input_dim:
            raise ValueError(f"Observation must be [batch,{self.input_dim}]")
        if obs.dtype != torch.float32 or not torch.isfinite(obs).all():
            raise ValueError("Observation must be finite float32")
        if state is None:
            state = self.initial_state(obs.shape[0], obs.device)
        if not isinstance(state, dict) or set(state) != set(self.components):
            raise ValueError("Graph recurrent state keys differ from configured nodes; use an explicit reset")
        features = {"observation": obs}
        updated, signals = {}, {}
        for node in self.nodes:
            name = node["id"]
            inputs = [features[source] for source in node["inputs"]]
            if len(inputs) == 1:
                value = inputs[0]
            elif node["merge"] == "concat":
                value = torch.cat(inputs, dim=-1)
            else:
                value = torch.stack(inputs, dim=0).sum(dim=0)
            component = self.components[name]
            result, next_state, instrumentation = component(value, state[name])
            if not isinstance(result, torch.Tensor) or tuple(result.shape) != (obs.shape[0], component.output_dim):
                raise ValueError(f"Component {name} returned incompatible output shape")
            if result.device != obs.device or result.dtype != obs.dtype or not torch.isfinite(result).all():
                raise ValueError(f"Component {name} returned incompatible dtype/device or nonfinite output")
            # Validate the returned state without adapting or dropping anything.
            component.validate(value, next_state)
            features[name], updated[name], signals[name] = result, next_state, instrumentation
        output = features[self.output_node]
        return {"logits": self.policy_head(output), "value": self.value_head(output).squeeze(-1),
                "state": updated, "signals": signals}

    def grow(self, node: str, add_count: int, optimizer: torch.optim.Optimizer, state: dict):
        if node not in self.components or not isinstance(self.components[node], ExpandableRIM):
            raise ValueError(f"Node {node!r} does not support expansion")
        if not isinstance(state, dict) or set(state) != set(self.components):
            raise ValueError("Growth requires full compatible graph state")
        component = self.components[node]
        extra = component.growth_parameter_count(add_count)
        if self.parameter_counts()["trainable"] + extra > self.max_parameters:
            raise ValueError("Growth exceeds graph trainable max_parameters")
        new_component_state, component_event = component.expand(add_count, optimizer, state[node])
        new_state = dict(state)
        new_state[node] = new_component_state
        event = {"node": node, **component_event, "trainable_parameters": self.parameter_counts()["trainable"]}
        self.growth_events.append(event)
        return new_state, event
