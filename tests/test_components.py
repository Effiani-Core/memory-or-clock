"""Component contracts, genuine gradients, selective updates and safe growth."""
from copy import deepcopy
from io import BytesIO
import pytest
import torch
from torch import nn
from ucl.components import (Component, MLP, GRU, LSTM, RIM, ExpandableRIM,
                            register_component, create_component, detach_state, reset_state)
from ucl.architecture import GraphPolicy
from ucl.growth import GrowthPolicy


def config(kind="lstm", **params):
    return {"nodes": [{"id": "encoder", "type": "mlp", "inputs": ["observation"], "params": {"output_dim": 8}},
                      {"id": "memory", "type": kind, "inputs": ["encoder"], "params": {"output_dim": 8, **params}}],
            "output": "memory"}


@pytest.mark.parametrize("kind", ["mlp", "gru", "lstm", "rim", "expandable_rim"])
def test_swappable_components_train_without_runner_changes(kind):
    torch.manual_seed(22)
    model = GraphPolicy(config(kind), 12, 6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    state = model.initial_state(3, "cpu")
    loss = 0
    for _ in range(3):
        result = model(torch.randn(3, 12), state)
        state = result["state"]
        loss = loss + nn.functional.cross_entropy(result["logits"], torch.tensor([0, 1, 2])) + result["value"].square().mean()
    optimizer.zero_grad()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert any(not torch.equal(before[name], p) for name, p in model.named_parameters() if name.startswith("components.memory"))
    assert result["logits"].shape == (3, 6)
    assert result["value"].shape == (3,)


def test_contract_has_required_declarations():
    for kind in ["mlp", "lstm", "gru", "rim", "expandable_rim"]:
        declaration = create_component(kind, 7).contract()
        assert {"input", "output", "state", "trainable_parameters", "frozen_parameters",
                "supported_devices", "learning_operations", "checkpoint", "structural_operations",
                "instrumentation", "external_memory"} <= declaration.keys()
        assert declaration["checkpoint"]["strict_shapes"]


def test_custom_component_registration_and_gradient_path():
    class CustomScale(Component):
        def __init__(self, input_dim, scale=2.0):
            super().__init__(input_dim, input_dim)
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        def forward(self, x, state):
            self.validate(x, state)
            return x * self.scale, {}, {"scale": self.scale.detach().item()}
    register_component("test_custom_scale", CustomScale)
    conf = {"nodes": [{"id": "custom", "type": "test_custom_scale", "inputs": ["observation"], "params": {"scale": 1.0}}], "output": "custom"}
    model = GraphPolicy(conf, 3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    result = model(torch.ones(2, 3))
    result["logits"].square().sum().backward()
    optimizer.step()
    assert model.components["custom"].scale.item() != 1.0
    with pytest.raises(ValueError, match="new and nonempty"):
        register_component("test_custom_scale", CustomScale)


def test_multibranch_explicit_composition():
    conf = {"nodes": [
        {"id": "a", "type": "lstm", "inputs": ["observation"], "params": {"output_dim": 3}},
        {"id": "b", "type": "gru", "inputs": ["observation"], "params": {"output_dim": 5}},
        {"id": "join", "type": "mlp", "inputs": ["a", "b"], "merge": "concat", "params": {"output_dim": 7}},
    ], "output": "join"}
    model = GraphPolicy(conf, 4, 6)
    assert model.components["join"].input_dim == 8
    assert model(torch.ones(2, 4))["logits"].shape == (2, 6)
    conf["nodes"][-1]["merge"] = "sum"
    with pytest.raises(ValueError, match="equal dimensions"):
        GraphPolicy(conf, 4, 6)
    conf["nodes"][-1].pop("merge")
    with pytest.raises(ValueError, match="explicit merge"):
        GraphPolicy(conf, 4, 6)


@pytest.mark.parametrize("error", ["unknown", "cycle", "state", "width", "dtype", "nonfinite"])
def test_invalid_inputs_and_configurations_are_rejected(error):
    conf = config()
    if error == "unknown":
        conf["nodes"][1]["type"] = "does_not_exist"
        with pytest.raises(ValueError, match="Unknown component"):
            GraphPolicy(conf, 12, 6)
    elif error == "cycle":
        conf["nodes"][0]["inputs"] = ["memory"]
        with pytest.raises(ValueError, match="cycles"):
            GraphPolicy(conf, 12, 6)
    else:
        model = GraphPolicy(conf, 12, 6)
        x = torch.zeros(2, 12)
        state = model.initial_state(2, "cpu")
        if error == "state":
            state["memory"]["h"] = torch.zeros(2, 7)
        elif error == "width":
            x = torch.zeros(2, 13)
        elif error == "dtype":
            x = x.double()
        elif error == "nonfinite":
            x[0, 0] = float("nan")
        with pytest.raises(ValueError):
            model(x, state)


def test_episode_reset_and_detach_preserve_other_batch_rows():
    model = GraphPolicy(config(), 12, 6)
    state = model(torch.randn(2, 12))["state"]
    detached = detach_state(state)
    assert not detached["memory"]["h"].requires_grad
    reset = reset_state(detached, torch.tensor([True, False]))
    for key in ["h", "c"]:
        assert not reset["memory"][key][0].any()
        assert torch.equal(reset["memory"][key][1], detached["memory"][key][1])
    with pytest.raises(ValueError):
        reset_state(state, torch.tensor([True]))


def test_rim_null_topk_inactive_identity_and_gradient():
    model = RIM(4, 4, modules=3, active_modules=1, attention_dim=2)
    with torch.no_grad():
        model.input_key.weight.fill_(1)
        for i, mechanism in enumerate(model.mechanisms):
            mechanism.input_query.weight.zero_()
            mechanism.input_query.bias.fill_(i + 1)
    h = torch.randn(2, 3, 4, requires_grad=True)
    output, state, signals = model(torch.ones(2, 4), {"h": h})
    active = torch.tensor(signals["active_mechanisms"])
    assert torch.equal(active, torch.tensor([[False, False, True]] * 2))
    assert torch.equal(state["h"][:, :2], h[:, :2])
    assert not torch.equal(state["h"][:, 2], h[:, 2])
    attention = torch.tensor(signals["input_attention"])
    assert torch.all(attention[:, 2, 0] < attention[:, 0, 0])
    state["h"][:, :2].sum().backward(retain_graph=True)
    assert torch.equal(h.grad[:, :2], torch.ones_like(h[:, :2]))
    model.zero_grad()
    output.square().sum().backward()
    assert model.mechanisms[2].cell.weight_ih.grad.abs().sum() > 0
    assert model.mechanisms[0].cell.weight_ih.grad is None
    assert model.mechanisms[1].cell.weight_ih.grad is None


def test_rim_communication_reads_inactive_without_modifying_it():
    model = RIM(3, 3, modules=2, active_modules=1, attention_dim=2)
    with torch.no_grad():
        model.input_key.weight.fill_(1)
        for i, mechanism in enumerate(model.mechanisms):
            mechanism.input_query.weight.zero_()
            mechanism.input_query.bias.fill_(i + 1)
            mechanism.comm_query.weight.zero_()
            mechanism.comm_key.weight.zero_()
            mechanism.comm_value.weight.copy_(torch.eye(3))
    first = {"h": torch.zeros(1, 2, 3)}
    second = {"h": first["h"].clone()}
    second["h"][:, 0] = 2.0
    _, state1, _ = model(torch.ones(1, 3), first)
    _, state2, _ = model(torch.ones(1, 3), second)
    assert torch.equal(state2["h"][:, 0], second["h"][:, 0])
    assert torch.allclose(state2["h"][:, 1] - state1["h"][:, 1], torch.ones(1, 3))


def test_growth_preserves_weights_optimizer_moments_and_old_state():
    torch.manual_seed(33)
    model = GraphPolicy(config("expandable_rim", modules=2, active_modules=1, max_modules=4), 12, 6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    result = model(torch.randn(3, 12))
    result["logits"].square().sum().backward()
    optimizer.step()
    state = detach_state(result["state"])
    old_parameters = {name: p for name, p in model.named_parameters()}
    old_values = {name: p.detach().clone() for name, p in old_parameters.items()}
    old_moments = {id(p): deepcopy(values) for p, values in optimizer.state.items()}
    before_count = model.parameter_counts()["trainable"]
    optimizer.param_groups[0]["lr"] = 0.003
    grown_state, event = model.grow("memory", 1, optimizer, state)
    for name, p in old_parameters.items():
        assert dict(model.named_parameters())[name] is p
        assert torch.equal(p, old_values[name])
        for key, value in old_moments.get(id(p), {}).items():
            assert torch.equal(optimizer.state[p][key], value)
    assert model.parameter_counts()["trainable"] > before_count
    assert torch.equal(grown_state["memory"]["h"][:, :2], state["memory"]["h"])
    assert not grown_state["memory"]["h"][:, 2].any()
    assert event["after_modules"] == 3
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[1]["lr"] == 0.003
    assert model.config["nodes"][1]["params"]["modules"] == 3
    with pytest.raises(ValueError, match="topology migration"):
        model(torch.randn(3, 12), state)
    optimizer.zero_grad()
    new_result = model(torch.randn(3, 12), grown_state)
    new_result["logits"].square().sum().backward()
    optimizer.step()
    assert any(p.grad is not None for p in optimizer.param_groups[1]["params"])


def test_growth_rejection_is_atomic_and_fixed_cannot_grow():
    model = GraphPolicy(config("expandable_rim", modules=2, active_modules=1, max_modules=2), 12, 6)
    optimizer = torch.optim.Adam(model.parameters())
    state = model.initial_state(1, "cpu")
    old = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="max_modules"):
        model.grow("memory", 1, optimizer, state)
    assert len(optimizer.param_groups) == 1
    assert all(torch.equal(old[k], v) for k, v in model.state_dict().items())
    fixed = GraphPolicy(config("rim"), 12, 6)
    with pytest.raises(ValueError, match="does not support"):
        fixed.grow("memory", 1, optimizer, fixed.initial_state(1, "cpu"))


def test_graph_budget_includes_heads_and_rejects_growth():
    conf = config("expandable_rim", modules=2, active_modules=1)
    model = GraphPolicy(conf, 12, 6)
    conf["max_parameters"] = model.parameter_counts()["trainable"]
    model = GraphPolicy(conf, 12, 6)
    with pytest.raises(ValueError, match="graph trainable max_parameters"):
        model.grow("memory", 1, torch.optim.Adam(model.parameters()), model.initial_state(1, "cpu"))


def test_expanded_serialization_exact_forward_and_optimizer_restore():
    torch.manual_seed(53)
    model = GraphPolicy(config("expandable_rim", modules=2, active_modules=1), 12, 6)
    optimizer = torch.optim.Adam(model.parameters())
    state, _ = model.grow("memory", 2, optimizer, model.initial_state(2, "cpu"))
    x = torch.randn(2, 12)
    result = model(x, state)
    result["logits"].square().sum().backward()
    optimizer.step()
    saved_names = {id(p): name for name, p in model.named_parameters()}
    group_names = [[saved_names[id(p)] for p in group["params"]] for group in optimizer.param_groups]
    checkpoint = {"config": model.config, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "groups": group_names, "state": detach_state(state)}
    stream = BytesIO()
    torch.save(checkpoint, stream)
    stream.seek(0)
    loaded = torch.load(stream, weights_only=True)
    restored = GraphPolicy(loaded["config"], 12, 6)
    restored.load_state_dict(loaded["model"], strict=True)
    parameters = dict(restored.named_parameters())
    new_optimizer = torch.optim.Adam([{"params": [parameters[name] for name in group]} for group in loaded["groups"]])
    new_optimizer.load_state_dict(loaded["optimizer"])
    assert torch.equal(model(x, state)["logits"], restored(x, loaded["state"])["logits"])
    assert len(new_optimizer.param_groups) == 2
    result = restored(x, loaded["state"])
    new_optimizer.zero_grad()
    result["logits"].square().sum().backward()
    new_optimizer.step()
    assert any(p.grad is not None for p in new_optimizer.param_groups[1]["params"])


def test_frozen_parameters_unchanged_while_downstream_learns():
    conf = config()
    conf["nodes"][0]["params"]["frozen"] = True
    model = GraphPolicy(conf, 12, 6)
    old = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.01)
    model(torch.randn(3, 12))["logits"].square().sum().backward()
    optimizer.step()
    assert all(torch.equal(old[name], p) for name, p in model.named_parameters() if name.startswith("components.encoder"))
    assert any(not torch.equal(old[name], p) for name, p in model.named_parameters() if name.startswith("components.memory"))
    assert model.parameter_counts()["frozen"] > 0
    assert model.components["encoder"].contract()["learning_operations"] == []


def test_growth_policies_and_serialization():
    none = GrowthPolicy(kind="none")
    assert not any(none.observe(i, 0.0) for i in range(1, 10))
    fixed = GrowthPolicy(kind="fixed_schedule", interval=2, cooldown=1, max_events=1)
    assert not fixed.observe(1, 0)
    assert fixed.observe(2, 0)
    fixed.record_growth(2)
    assert not fixed.observe(4, 0)
    plateau = GrowthPolicy(kind="plateau", patience=4, cooldown=1, min_improvement=0.1)
    assert not any(plateau.observe(i, 1.0) for i in range(1, 4))
    restored = GrowthPolicy.from_state_dict(plateau.state_dict())
    assert restored.observe(4, 1.0)
    restored.record_growth(4)
    assert restored.events == 1 and not restored.history
    with pytest.raises(ValueError, match="strictly increasing"):
        restored.observe(4, 1.0)
