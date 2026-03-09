"""Utilities for automatic mixed precision (AMP)."""

from __future__ import annotations

from contextlib import nullcontext

import torch


class AMPScaler:
    """Small wrapper around PyTorch AMP scaler/autocast.

    AMP is automatically disabled on CPU-only runs.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled and torch.cuda.is_available())

        # Prefer modern torch.amp API, with fallback for older versions.
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled)
            self._autocast = lambda: torch.amp.autocast(device_type="cuda", enabled=self.enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.enabled)
            self._autocast = lambda: torch.cuda.amp.autocast(enabled=self.enabled)

    def autocast(self):
        if not self.enabled:
            return nullcontext()
        return self._autocast()

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return loss
        return self.scaler.scale(loss)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if not self.enabled:
            optimizer.step()
            return
        self.scaler.step(optimizer)

    def update(self) -> None:
        if self.enabled:
            self.scaler.update()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        if self.enabled:
            self.scaler.unscale_(optimizer)

    def state_dict(self):
        if not self.enabled:
            return {}
        return self.scaler.state_dict()

    def load_state_dict(self, state_dict):
        if self.enabled:
            self.scaler.load_state_dict(state_dict)

    @property
    def is_enabled(self) -> bool:
        return self.enabled


def initialize_scaler(enabled: bool = True) -> AMPScaler:
    return AMPScaler(enabled=enabled)


def scale_loss(loss: torch.Tensor, scaler: AMPScaler | None) -> torch.Tensor:
    if scaler is None:
        return loss
    return scaler.scale(loss)


def step_optimizer(optimizer: torch.optim.Optimizer, scaler: AMPScaler | None) -> None:
    if scaler is None:
        optimizer.step()
        return
    scaler.step(optimizer)


def update_scaler(scaler: AMPScaler | None) -> None:
    if scaler is not None:
        scaler.update()
