"""
Central training state container.

Holds persistent runtime state required for training
and checkpoint restoration.
"""

import torch

from stackformer.utils.utils import get_rank, get_world_size


class TrainingState:
    """
    Container for all runtime training state.

    Responsibilities
    ----------------
    • Store model / optimizer / scheduler
    • Track training progress (epoch, step, batch)
    • Store experiment configuration
    • Provide helper utilities for device and LR
    • Support checkpoint restoration
    • Provide distributed training info
    """

    def __init__(
        self,
        model=None,
        optimizer=None,
        scheduler=None,
        scaler=None,
        device=None,
        config=None,
    ):

        # -------------------------------------------------
        # Device handling
        # -------------------------------------------------

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        # -------------------------------------------------
        # Core components
        # -------------------------------------------------

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

        # -------------------------------------------------
        # Training progress
        # -------------------------------------------------

        self.epoch = 0
        self.global_step = 0
        self.batch_idx = 0

        # -------------------------------------------------
        # Experiment config
        # -------------------------------------------------

        self.config = config or {}

        # -------------------------------------------------
        # Move model to device if provided
        # -------------------------------------------------

        if self.model is not None:
            self.model.to(self.device)

    # -----------------------------------------------------
    # Step update
    # -----------------------------------------------------

    def step(self):
        """
        Called after optimizer step.
        """
        self.global_step += 1

    # -----------------------------------------------------

    def next_epoch(self):
        """
        Advance training epoch.
        """
        self.epoch += 1
        self.batch_idx = 0

    # -----------------------------------------------------

    def reset_batch(self):
        """
        Reset batch index when starting new epoch.
        """
        self.batch_idx = 0

    # -----------------------------------------------------
    # Learning rate helper
    # -----------------------------------------------------

    def get_lr(self):
        """
        Get current learning rate from optimizer.
        """

        if self.optimizer is None:
            return None

        return self.optimizer.param_groups[0]["lr"]

    # -----------------------------------------------------
    # Device helper
    # -----------------------------------------------------

    def to_device(self, obj):
        """
        Move tensor or batch to training device.
        """

        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)

        if isinstance(obj, dict):
            return {k: self.to_device(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return type(obj)(self.to_device(x) for x in obj)

        return obj

    # -----------------------------------------------------
    # Checkpoint metadata export
    # -----------------------------------------------------

    def metadata(self):
        """
        Return metadata used for checkpoint saving.
        """

        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "batch_idx": self.batch_idx,
        }

    # -----------------------------------------------------
    # Restore checkpoint metadata
    # -----------------------------------------------------

    def load_metadata(self, metadata):
        """
        Restore training progress from checkpoint.
        """

        self.epoch = metadata.get("epoch", 0)
        self.global_step = metadata.get("global_step", 0)
        self.batch_idx = metadata.get("batch_idx", 0)

    # -----------------------------------------------------
    # AMP helper
    # -----------------------------------------------------

    @property
    def amp_enabled(self):
        """
        Whether AMP training is enabled.
        """

        if self.scaler is None:
            return False

        return getattr(self.scaler, "is_enabled", False)

    # -----------------------------------------------------
    # Distributed helpers
    # -----------------------------------------------------

    @property
    def rank(self):
        """
        Current process rank.
        """
        return get_rank()

    # -----------------------------------------------------

    @property
    def world_size(self):
        """
        Total number of processes.
        """
        return get_world_size()

    # -----------------------------------------------------

    @property
    def is_distributed(self):
        """
        Whether distributed training is active.
        """
        return self.world_size > 1

    # -----------------------------------------------------

    @property
    def is_main_process(self):
        """
        True only for rank 0 process.
        """
        return self.rank == 0