"""Distributed training example.

Single-process CPU run:
    python examples/train_ddp.py

Multi-process CPU run:
    torchrun --standalone --nproc_per_node=2 examples/train_ddp.py
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stackformer.distributed import cleanup_distributed
from stackformer.engine import Trainer


def main():
    inputs = torch.randint(0, 8, (32, 4))
    targets = torch.randint(0, 8, (32, 4))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)

    model = nn.Sequential(nn.Embedding(8, 16), nn.Linear(16, 8))
    trainer = Trainer(
        model=model,
        train_dataloader=loader,
        device="cpu",
        use_ddp=True,
        ddp_backend="gloo",
        max_epochs=1,
        max_train_steps=2,
        checkpoint_dir="ddp_example_ckpts",
    )
    trainer.fit()
    cleanup_distributed()
    print("DDP example finished.")


if __name__ == "__main__":
    main()
