"""Device detection and movement helpers."""

from __future__ import annotations

import torch

from stackformer.utils.utils import is_main_process, print_once


def get_device(device=None):
    """Resolve a training device.

    Priority:
      1. Explicit ``device`` argument.
      2. CUDA when available.
      3. MPS when available.
      4. CPU fallback.
    """
    if device is not None and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(obj, device):
    """Recursively move tensors/collections to device."""
    if torch.is_tensor(obj):
        non_blocking = str(device).startswith("cuda")
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)
    return obj


def format_memory(bytes_value: int) -> str:
    return f"{bytes_value / (1024 ** 3):.2f} GB"


def print_device_info() -> None:
    """Print runtime device summary from rank-0 only."""
    if not is_main_process():
        return

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print_once(f"Device: CUDA:{idx} {props.name}")
        print_once(f"GPU Memory: {format_memory(props.total_memory)}")
        return

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print_once("Device: MPS")
        return

    print_once("Device: CPU")


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
