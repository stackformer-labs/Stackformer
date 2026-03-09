"""Training state container."""

from __future__ import annotations

import torch

from stackformer.utils.utils import get_rank, get_world_size


class TrainingState:
    """Container for mutable training runtime state."""

    def __init__(self, model=None, optimizer=None, scheduler=None, scaler=None, device=None, config=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

        self.epoch = 0
        self.global_step = 0
        self.batch_idx = 0
        self.config = config or {}

        if self.model is not None:
            self.model.to(self.device)

    def step(self) -> None:
        self.global_step += 1

    def next_epoch(self) -> None:
        self.epoch += 1
        self.batch_idx = 0

    def reset_batch(self) -> None:
        self.batch_idx = 0

    def get_lr(self):
        if self.optimizer is None:
            return None
        return self.optimizer.param_groups[0]["lr"]

    def to_device(self, obj):
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self.to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.to_device(x) for x in obj)
        return obj

    def metadata(self):
        return {"epoch": self.epoch, "global_step": self.global_step, "batch_idx": self.batch_idx}

    def load_metadata(self, metadata):
        self.epoch = metadata.get("epoch", 0)
        self.global_step = metadata.get("global_step", 0)
        self.batch_idx = metadata.get("batch_idx", 0)

    @property
    def amp_enabled(self):
        return bool(self.scaler is not None and getattr(self.scaler, "is_enabled", False))

    @property
    def rank(self):
        return get_rank()

    @property
    def world_size(self):
        return get_world_size()

    @property
    def is_distributed(self):
        return self.world_size > 1

    @property
    def is_main_process(self):
        return self.rank == 0
