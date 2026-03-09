"""Utilities for automatic mixed precision (AMP).

This module wraps ``torch.cuda.amp.GradScaler`` behind a small interface that
is safe across CPU/CUDA environments.
"""

import torch
from contextlib import nullcontext


class AMPScaler:
    """Automatic mixed precision scaler wrapper.

    Args:
        enabled: Whether AMP should be enabled when CUDA is available.
    """

    def __init__(self, enabled: bool = True):

        self.enabled = enabled and torch.cuda.is_available()

        if self.enabled:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def autocast(self):
        """Return an autocast context manager.

        Returns:
            A CUDA autocast context when AMP is enabled, else a no-op context.
        """
        if not self.enabled:
            return nullcontext()
        return torch.cuda.amp.autocast()

    # -------------------------------------------------------------

    def scale(self, loss):
        """
        Scale the loss for AMP training.
        """
        if not self.enabled:
            return loss

        return self.scaler.scale(loss)

    # -------------------------------------------------------------

    def step(self, optimizer):
        """
        Step optimizer safely with scaler.
        """
        if not self.enabled:
            optimizer.step()
            return

        self.scaler.step(optimizer)

    # -------------------------------------------------------------

    def update(self):
        """
        Update scaler after optimizer step.
        """
        if not self.enabled:
            return

        self.scaler.update()

    # -------------------------------------------------------------

    def unscale_(self, optimizer):
        """
        Unscale gradients before gradient clipping.
        """
        if not self.enabled:
            return

        self.scaler.unscale_(optimizer)

    # -------------------------------------------------------------

    @property
    def is_enabled(self):
        """
        Whether AMP is active.
        """
        return self.enabled
