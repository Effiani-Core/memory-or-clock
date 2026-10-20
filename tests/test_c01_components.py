"""C01 controls: the episode clock, the frame stack and the LSTM whose memory is erased."""

import pytest
import torch

from ucl.architecture import GraphPolicy
from ucl.cli import ROOT, read_config
from ucl.components import create_component


def test_clock_counts_steps_and_restarts_with_the_episode():
    clock = create_component("clock", input_dim=2, horizon=4)
    state = clock.initial_state(1, "cpu")
    times = []
    for _ in range(3):
        out, state, _ = clock(torch.ones(1, 2), state)
        times.append(out[0, -1].item())
    assert times == [0.0, 0.25, 0.5] and out.shape == (1, 3)
    assert clock.initial_state(1, "cpu")["t"].item() == 0
    assert not list(clock.parameters())


def test_frame_stack_keeps_the_last_frames_oldest_first():
    stack = create_component("frame_stack", input_dim=1, frames=3)
    state = stack.initial_state(1, "cpu")
    outputs = []
    for value in (1.0, 2.0, 3.0, 4.0):
        out, state, _ = stack(torch.tensor([[value]]), state)
        outputs.append(out[0].tolist())
    assert outputs == [[0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]]
    with pytest.raises(ValueError, match="frames"):
        create_component("frame_stack", input_dim=1, frames=1)


def test_lstm_reset_has_the_lstm_parameters_and_ignores_its_past():
    lstm = create_component("lstm", input_dim=3, output_dim=4)
    reset = create_component("lstm_reset", input_dim=3, output_dim=4)
    reset.load_state_dict(lstm.state_dict())
    x = torch.randn(1, 3)
    fresh, _, _ = reset(x, reset.initial_state(1, "cpu"))
    busy = {"h": torch.randn(1, 4), "c": torch.randn(1, 4)}
    after_history, next_state, _ = reset(x, busy)
    assert torch.equal(fresh, after_history)
    assert not next_state["h"].any() and not next_state["c"].any()
    remembering, _, _ = lstm(x, busy)
    assert not torch.equal(remembering, fresh)


@pytest.mark.parametrize("name, low, high", [("MLP-wide", 11367, 11400),
                                             ("stack4", 11367, 11400),
                                             ("LSTM-reset", 11367, 11367)])
def test_c01_arms_match_the_lstm_size(name, low, high):
    config = read_config(ROOT / f"configs/study_001/arch-{name}.json")["architecture"]
    model = GraphPolicy(config, 50, 6)
    assert low <= model.parameter_counts()["trainable"] <= high
    output = model(torch.zeros(2, 50))
    assert output["logits"].shape == (2, 6)


def test_the_wide_mlp_without_clock_matches_the_lstm_size():
    config = read_config(ROOT / "configs/study_001/arch-MLP-wide-noclock.json")["architecture"]
    model = GraphPolicy(config, 50, 6)
    assert 11367 <= model.parameter_counts()["trainable"] <= 11410
    assert "clock" not in {node["type"] for node in config["nodes"]}


def test_the_mechanism_arms_know_the_time_and_have_no_memory():
    reset = read_config(ROOT / "configs/study_001/arch-LSTM-reset-clock.json")["architecture"]
    small = read_config(ROOT / "configs/study_001/arch-MLP-clock.json")["architecture"]
    assert GraphPolicy(reset, 50, 6).parameter_counts()["trainable"] == 11495
    assert GraphPolicy(small, 50, 6).parameter_counts()["trainable"] == 2951
    for config in (reset, small):
        types = {node["type"] for node in config["nodes"]}
        assert "clock" in types and "lstm" not in types
