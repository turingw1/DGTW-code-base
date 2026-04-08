from __future__ import annotations

import copy

import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        shadow_params = dict(self.shadow.named_parameters())
        model_params = dict(model.named_parameters())
        for name, shadow in shadow_params.items():
            shadow.lerp_(model_params[name].detach(), 1.0 - self.decay)

        shadow_buffers = dict(self.shadow.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, shadow in shadow_buffers.items():
            shadow.copy_(model_buffers[name].detach())

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow.load_state_dict(state_dict)
