import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tests._test_utils import _checkpoint

from stackformer.engine import Trainer


class TinyTokenModel(nn.Module):
    def __init__(self, vocab_size=13, hidden=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.proj(self.embedding(x))


def _eval_loss(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            losses.append(criterion(model(x).view(-1, 13), y.view(-1)).item())
    return sum(losses) / len(losses)


def test_built_in_training_runs_and_loss_is_reasonable(tmp_path, torch_device):
    _checkpoint("test_built_in_training_runs_and_loss_is_reasonable setup", device=torch_device)
    torch.manual_seed(7)
    x = torch.randint(0, 13, (48, 6))
    y = x.clone()
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)

    model = TinyTokenModel().to(torch_device)
    before = _eval_loss(model, loader, torch_device)

    _checkpoint("Fitting Trainer on TinyTokenModel", loss_before=before)
    trainer = Trainer(
        model=model,
        train_dataloader=loader,
        val_dataloader=loader,
        device=str(torch_device),
        max_epochs=3,
        checkpoint_dir=str(tmp_path),
    )
    trainer.fit()
    after = _eval_loss(model, loader, torch_device)

    _checkpoint("Asserting global_step and loss reduction", global_step=trainer.state.global_step, loss_after=after)
    assert trainer.state.global_step > 0
    assert after <= before
