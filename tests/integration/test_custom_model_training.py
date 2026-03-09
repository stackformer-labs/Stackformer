import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stackformer.engine import Trainer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x)


def test_custom_model_trains_with_stackformer_trainer(tmp_path):
    torch.manual_seed(42)
    x = torch.randn(64, 4)
    y = (2 * x[:, :1] - x[:, 1:2] + 0.1).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)

    model = SimpleModel()
    criterion = nn.MSELoss()

    with torch.no_grad():
        before = criterion(model(x), y).item()

    trainer = Trainer(
        model=model,
        train_dataloader=loader,
        val_dataloader=loader,
        criterion=criterion,
        device="cpu",
        max_epochs=3,
        checkpoint_dir=str(tmp_path),
    )
    trainer.fit()

    with torch.no_grad():
        after = criterion(model(x), y).item()

    assert after < before
    assert (tmp_path / "checkpoint_latest.pt").exists()
