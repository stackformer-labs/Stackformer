"""Minimal CPU training example using StackFormer Trainer."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stackformer.engine import Trainer


def build_model(vocab_size: int = 16, hidden: int = 32):
    return nn.Sequential(nn.Embedding(vocab_size, hidden), nn.Linear(hidden, vocab_size))


def main():
    torch.manual_seed(0)
    inputs = torch.randint(0, 16, (64, 6))
    targets = torch.randint(0, 16, (64, 6))
    train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=8, shuffle=True)

    model = build_model()
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        device="cpu",
        max_epochs=1,
        max_train_steps=3,
        checkpoint_dir="example_ckpts",
    )
    trainer.fit()
    print("Simple train finished on CPU.")


if __name__ == "__main__":
    main()
