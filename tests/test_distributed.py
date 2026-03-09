import os

import torch

from stackformer.distributed import barrier, cleanup_distributed, init_distributed, is_main_process


def test_distributed_gloo_single_process_init():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"

    initialized = init_distributed(backend="gloo")
    assert initialized is False
    assert is_main_process()
    barrier()
    cleanup_distributed()
    assert torch.distributed.is_initialized() is False
