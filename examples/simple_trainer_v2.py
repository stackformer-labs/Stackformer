"""Example showing modular V2 trainer components."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stackformer.engine import Engine, TrainingState
from stackformer.optim import create_optimizer


def main():
    torch.manual_seed(1)
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
    optimizer = create_optimizer(model, optimizer_name="adam", lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    state = TrainingState(model=model, optimizer=optimizer, device="cpu", config={"criterion": criterion})
    engine = Engine(state=state, max_train_steps=2)
    engine.train_one_epoch(loader, epoch=0)
    print("V2 modular engine run complete.")


if __name__ == "__main__":
    main()
