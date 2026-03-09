"""
Distributed training utilities.

Provides a clean wrapper around PyTorch DistributedDataParallel (DDP)
so the rest of the framework remains backend-agnostic.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# -----------------------------------------------------
# Initialization
# -----------------------------------------------------

def init_distributed(backend=None):
    """
    Initialize distributed process group.

    Works with:
    torchrun
    torch.distributed.launch
    SLURM clusters
    """

    if dist.is_initialized():
        return

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Environment variables set by torchrun
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())


# -----------------------------------------------------
# Rank helpers
# -----------------------------------------------------

def get_rank():
    """
    Return process rank.
    """
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """
    Return total number of processes.
    """
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank():
    """
    Rank within node.
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def is_distributed():
    """
    Whether distributed training is active.
    """
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    """
    True only for rank 0 process.
    """
    return get_rank() == 0


# -----------------------------------------------------
# Model wrapping
# -----------------------------------------------------

def wrap_model_ddp(model):
    """
    Wrap model with DistributedDataParallel.
    """

    if not is_distributed():
        return model

    device = torch.device("cuda", get_local_rank()) if torch.cuda.is_available() else torch.device("cpu")

    model = model.to(device)

    model = DDP(
        model,
        device_ids=[get_local_rank()] if torch.cuda.is_available() else None,
        output_device=get_local_rank() if torch.cuda.is_available() else None,
        find_unused_parameters=False,
    )

    return model


# -----------------------------------------------------
# DataLoader utilities
# -----------------------------------------------------

def distributed_sampler(dataset, shuffle=True):
    """
    Create DistributedSampler for dataset.
    """

    if not is_distributed():
        return None

    from torch.utils.data.distributed import DistributedSampler

    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
    )


# -----------------------------------------------------
# Synchronization utilities
# -----------------------------------------------------

def barrier():
    """
    Synchronize all processes.
    """

    if not is_distributed():
        return

    dist.barrier()


def cleanup():
    """
    Destroy distributed process group.
    """

    if not is_distributed():
        return

    dist.destroy_process_group()