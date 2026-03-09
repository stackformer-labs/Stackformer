"""CPU-default distributed training setup example."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stackformer.distributed import init_distributed, is_distributed
from stackformer.engine import Trainer


def main():
    # No-op in single-process CPU runs, but demonstrates DDP entrypoint.
    init_distributed(backend="gloo")

    inputs = torch.randint(0, 8, (32, 4))
    targets = torch.randint(0, 8, (32, 4))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)

    model = nn.Sequential(nn.Embedding(8, 16), nn.Linear(16, 8))
    trainer = Trainer(
        model=model,
        train_dataloader=loader,
        device="cpu",
        use_ddp=is_distributed(),
        max_epochs=1,
        max_train_steps=2,
        checkpoint_dir="ddp_example_ckpts",
    )
    trainer.fit()
    print("DDP example finished (single-process compatible).")


if __name__ == "__main__":
    main()
