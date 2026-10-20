"""Strict resume: topology, named optimizer groups, recurrent state and RNG."""
from __future__ import annotations
import json
import os
import random
from pathlib import Path
import torch
import numpy as np
from .architecture import GraphPolicy
from .frozen import FrozenLLMPolicy, register_frozen


def build_policy(config, input_dim, action_count, device="cpu"):
    register_frozen()
    kind = config.get("kind", "graph")
    if kind == "frozen_llm_baseline":
        model = FrozenLLMPolicy(config, input_dim, action_count)
    elif kind == "graph":
        model = GraphPolicy(config, input_dim, action_count)
    else:
        raise ValueError(f"Unknown policy kind: {kind}")
    return model.to(device)


def external_references(model):
    references, prefixes = {}, []
    for name, module in model.named_modules():
        if hasattr(module, "external_checkpoint_reference"):
            reference = module.external_checkpoint_reference()
            references[name] = reference
            prefixes += [name + "." + prefix if name else prefix for prefix in reference["prefixes"]]
    return references, prefixes


def save_checkpoint(path, model, trainer=None, *, metadata=None, growth=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    references, prefixes = external_references(model)
    weights = {key: value.detach().cpu() for key, value in model.state_dict().items()
               if not any(key.startswith(prefix) for prefix in prefixes)}
    names = {id(p): name for name, p in model.named_parameters()}
    numpy_rng = np.random.get_state()
    payload = {"format_version": 1,
               "numpy_rng": {"algorithm": numpy_rng[0], "keys": numpy_rng[1].tolist(),
                             "position": numpy_rng[2], "has_gauss": numpy_rng[3], "cached_gaussian": numpy_rng[4]}, "architecture": model.config,
               "input_dim": model.input_dim, "action_count": model.action_count,
               "weights": weights, "external_references": references,
               "metadata": metadata or {}, "growth_events": model.growth_events,
               "growth_policy": growth.state_dict() if growth is not None else None,
               "trainer": trainer.state_dict() if trainer else None,
               "optimizer": trainer.optimizer.state_dict() if trainer else None,
               "optimizer_parameter_names": [[names[id(p)] for p in group["params"]]
                                             for group in trainer.optimizer.param_groups] if trainer else None,
               "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(path, device="cpu", expected_architecture=None):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    if expected_architecture is not None and json.dumps(expected_architecture, sort_keys=True) != json.dumps(payload["architecture"], sort_keys=True):
        raise ValueError("Checkpoint architecture differs from expected topology")
    model = build_policy(payload["architecture"], payload["input_dim"], payload["action_count"], device)
    references, prefixes = external_references(model)
    if references != payload["external_references"]:
        raise ValueError("Frozen model reference changed; checkpoint is incompatible")
    expected = set(model.state_dict())
    allowed_missing = {key for key in expected if any(key.startswith(prefix) for prefix in prefixes)}
    if set(payload["weights"]) != expected - allowed_missing:
        raise ValueError("Checkpoint parameter names differ from architecture")
    incompatible = model.load_state_dict(payload["weights"], strict=False)
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise ValueError("Checkpoint failed strict parameter compatibility")
    model.growth_events = payload["growth_events"]
    optimizer = None
    if payload["optimizer"] is not None:
        named = dict(model.named_parameters())
        groups = []
        for names, saved in zip(payload["optimizer_parameter_names"], payload["optimizer"]["param_groups"], strict=True):
            if any(name not in named for name in names):
                raise ValueError("Checkpoint optimizer refers to unknown parameter")
            group = {k: v for k, v in saved.items() if k != "params"}
            group["params"] = [named[name] for name in names]
            groups.append(group)
        optimizer = torch.optim.Adam(groups)
        optimizer.load_state_dict(payload["optimizer"])
    return model, optimizer, payload


def restore_rng(payload):
    torch.set_rng_state(payload["torch_rng"])
    random.setstate(payload["python_rng"])
    if "numpy_rng" in payload:
        state = payload["numpy_rng"]
        np.random.set_state((state["algorithm"], np.asarray(state["keys"], dtype=np.uint32),
                             state["position"], state["has_gauss"], state["cached_gaussian"]))
    if payload["cuda_rng"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG checkpoint requires CUDA to resume exactly")
        torch.cuda.set_rng_state_all(payload["cuda_rng"])


def validate_selected_architecture(selected, restored):
    """Allow documented expansion, reject selecting a different architecture silently."""
    import copy
    left, right = copy.deepcopy(selected), copy.deepcopy(restored)
    if left.get("kind", "graph") != right.get("kind", "graph"):
        raise ValueError("Selected architecture differs from checkpoint")
    if left.get("kind", "graph") == "graph":
        left_nodes, right_nodes = left.get("nodes", []), right.get("nodes", [])
        if len(left_nodes) != len(right_nodes):
            raise ValueError("Selected graph differs from checkpoint")
        for a, b in zip(left_nodes, right_nodes, strict=True):
            if a.get("type") == b.get("type") == "expandable_rim":
                requested = a.get("params", {})
                actual = b.get("params", {})
                count = actual.get("modules", 2)
                if not requested.get("modules",2) <= count <= requested.get("max_modules",8):
                    raise ValueError("Checkpoint topology exceeds selected expansion limits")
                requested.pop("modules", None)
                actual.pop("modules", None)
    if left != right:
        raise ValueError("Selected architecture configuration differs from checkpoint")
