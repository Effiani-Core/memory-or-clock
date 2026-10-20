"""Example plugin. Activate with config plugins: ["ucl.custom_example"]."""
import torch
from .components import Component, REGISTRY, register_component

class CustomTanh(Component):
    def __init__(self, input_dim, output_dim=32):
        super().__init__(input_dim, output_dim)
        self.projection = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x, state):
        self.validate(x, state)
        result = torch.tanh(self.projection(x))
        return result, {}, {"activation_mean": result.detach().mean().item()}

if "custom_tanh" not in REGISTRY:
    register_component("custom_tanh", CustomTanh)
