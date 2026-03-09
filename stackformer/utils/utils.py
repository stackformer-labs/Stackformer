"""General utility helpers used across StackFormer."""

from __future__ import annotations

import os
import random
import time
from typing import Any

import torch

try:
    import numpy as np
except Exception:  # optional dependency in minimal environments
    np = None


def seed_everything(seed: int = 42) -> None:
    """Set reproducibility seeds across supported libraries."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_to_device(obj: Any, device):
    """Recursively move tensors and collections to ``device``."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)
    return obj


def ensure_dir(path: str) -> str:
    """Create a directory if it does not exist and return the same path."""
    os.makedirs(path, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def timestamp() -> str:
    """Return unix timestamp string."""
    return str(int(time.time()))


# Distributed-safe helpers

def get_rank() -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size() -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def print_once(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)
