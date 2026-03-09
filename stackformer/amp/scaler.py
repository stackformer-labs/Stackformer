"""
AMP management utilities.

Provides a thin wrapper around torch.cuda.amp.GradScaler
to keep mixed precision logic isolated from the training engine.
"""

import torch


class AMPScaler:
    """
    Automatic Mixed Precision scaler wrapper.

    Handles:
    - loss scaling
    - optimizer stepping
    - scaler updates
    - safe CPU fallback
    """

    def __init__(self, enabled: bool = True):

        self.enabled = enabled and torch.cuda.is_available()

        if self.enabled:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

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