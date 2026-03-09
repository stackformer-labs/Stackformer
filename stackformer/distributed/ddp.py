"""DistributedDataParallel helpers."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def init_distributed(backend: str | None = None) -> None:
    """Initialize torch.distributed when launched in distributed mode.

    This function safely no-ops in single-process environments.
    """
    if not dist.is_available() or dist.is_initialized():
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return

    rank = int(os.environ.get("RANK", "0"))
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())


def get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return get_rank() == 0


def wrap_model_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """Wrap model with DDP when distributed is initialized."""
    if not is_distributed():
        return model

    if torch.cuda.is_available():
        local_rank = get_local_rank()
        model = model.to(torch.device("cuda", local_rank))
        return DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    return DDP(model, device_ids=None, output_device=None, find_unused_parameters=False)


def distributed_sampler(dataset, shuffle: bool = True):
    """Create ``DistributedSampler`` only when in distributed mode."""
    if not is_distributed():
        return None
    from torch.utils.data.distributed import DistributedSampler

    return DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=shuffle)


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def cleanup() -> None:
    if is_distributed():
        dist.destroy_process_group()


def setup_ddp(backend: str | None = None) -> None:
    """Backward-compatible alias for distributed initialization."""
    init_distributed(backend=backend)
