"""
General utilities used across StackFormer.
"""

import os
import random
import time
from typing import Any

import numpy as np
import torch


# -----------------------------------------------------
# Reproducibility
# -----------------------------------------------------

def seed_everything(seed: int = 42):
    """
    Set random seed for full reproducibility.

    Affects:
    - Python
    - NumPy
    - PyTorch
    - CUDA
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------
# Device utilities
# -----------------------------------------------------

def move_to_device(obj: Any, device):
    """
    Recursively move tensors to device.

    Supports:
    - tensors
    - dict
    - list / tuple
    """

    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)

    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)

    return obj


# -----------------------------------------------------
# Model utilities
# -----------------------------------------------------

def count_parameters(model, trainable_only: bool = True):
    """
    Count number of model parameters.
    """

    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters())


def format_parameters(num_params: int):
    """
    Format parameter count into readable string.
    """

    if num_params >= 1e9:
        return f"{num_params/1e9:.2f}B"

    if num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"

    if num_params >= 1e3:
        return f"{num_params/1e3:.2f}K"

    return str(num_params)


# -----------------------------------------------------
# Timing utilities
# -----------------------------------------------------

def format_time(seconds: float):
    """
    Convert seconds into readable time string.
    """

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"

    if minutes > 0:
        return f"{minutes}m {seconds}s"

    return f"{seconds}s"


def current_time():
    """
    Return formatted timestamp.
    """

    return time.strftime("%Y-%m-%d %H:%M:%S")


# -----------------------------------------------------
# Distributed helpers
# -----------------------------------------------------

def get_rank():
    """
    Return process rank safely.
    """

    if not torch.distributed.is_available():
        return 0

    if not torch.distributed.is_initialized():
        return 0

    return torch.distributed.get_rank()


def get_world_size():
    """
    Return world size safely.
    """

    if not torch.distributed.is_available():
        return 1

    if not torch.distributed.is_initialized():
        return 1

    return torch.distributed.get_world_size()


def is_main_process():
    """
    True only for rank 0 process.
    """

    return get_rank() == 0


# -----------------------------------------------------
# Logging helpers
# -----------------------------------------------------

def print_once(*args, **kwargs):
    """
    Print only from main process.
    """

    if is_main_process():
        print(*args, **kwargs)