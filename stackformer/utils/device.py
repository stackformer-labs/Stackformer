"""
Device utilities for StackFormer.

Handles:
- device auto detection
- GPU information
- safe device movement
- memory reporting
"""

import torch

from stackformer.utils.utils import is_main_process, print_once


# -----------------------------------------------------
# Device selection
# -----------------------------------------------------

def get_device(device=None):
    """
    Return torch.device safely.

    Priority:
    1) user provided device
    2) CUDA if available
    3) CPU fallback
    """

    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# -----------------------------------------------------
# GPU information
# -----------------------------------------------------

def get_gpu_name():
    """
    Return GPU name.
    """

    if not torch.cuda.is_available():
        return None

    return torch.cuda.get_device_name(0)


def get_gpu_memory():
    """
    Return GPU memory info.
    """

    if not torch.cuda.is_available():
        return None

    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)

    return {
        "total": total,
        "reserved": reserved,
        "allocated": allocated,
        "free": reserved - allocated,
    }


def format_memory(bytes_value):
    """
    Format bytes to readable memory units.
    """

    gb = bytes_value / (1024 ** 3)

    return f"{gb:.2f} GB"


# -----------------------------------------------------
# Device reporting
# -----------------------------------------------------

def print_device_info():
    """
    Print device information.
    Only prints from main process in DDP.
    """

    if not is_main_process():
        return

    if torch.cuda.is_available():

        gpu_name = get_gpu_name()
        memory = torch.cuda.get_device_properties(0).total_memory

        print_once("Device:", gpu_name)
        print_once("GPU Memory:", format_memory(memory))

    else:

        print_once("Device: CPU")


# -----------------------------------------------------
# Tensor helpers
# -----------------------------------------------------

def move_to_device(obj, device):
    """
    Move tensors / collections to device.
    """

    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)

    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)

    return obj


# -----------------------------------------------------
# CUDA utilities
# -----------------------------------------------------

def clear_cuda_cache():
    """
    Clear GPU memory cache.
    """

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize():
    """
    Synchronize CUDA operations.
    Useful for benchmarking.
    """

    if torch.cuda.is_available():
        torch.cuda.synchronize()