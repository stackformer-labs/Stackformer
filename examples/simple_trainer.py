"""Example script demonstrating modular Trainer API execution.

Constructs a synthetic dataset and trains a basic embedding-linear model for 1 epoch.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stackformer.engine import Trainer


def main() -> None:
    """Run simple trainer example."""
    torch.manual_seed(1)
    x = torch.randint(0, 8, (64, 5))
    y = torch.randint(0, 8, (64, 5))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    model = nn.Sequential(nn.Embedding(8, 16), nn.Linear(16, 8))
    trainer = Trainer(
        model=model,
        train_dataloader=loader,
        val_dataloader=loader,
        device="cpu",
        use_amp=True,
        use_ddp=False,
        max_epochs=1,
        max_train_steps=3,
        max_eval_steps=1,
        checkpoint_dir="example_v3_ckpts",
    )
    trainer.fit()
    print("V3 modular trainer run complete.")


if __name__ == "__main__":
    main()

