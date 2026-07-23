import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from tests._test_utils import _checkpoint

from stackformer.engine import Trainer


class TinyLM(nn.Module):
    def __init__(self, vocab_size=11, hidden=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.proj(self.embedding(x))


def test_trainer_loop_executes_with_amp_on_cpu(tmp_path, torch_device):
    _checkpoint("test_trainer_loop_executes_with_amp_on_cpu setup", device=torch_device)
    x = torch.randint(0, 11, (20, 5))
    y = torch.randint(0, 11, (20, 5))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    trainer = Trainer(
        model=TinyLM().to(torch_device),
        train_dataloader=loader,
        val_dataloader=loader,
        device=str(torch_device),
        use_amp=True,
        max_epochs=1,
        max_train_steps=3,
        max_eval_steps=1,
        checkpoint_dir=str(tmp_path),
    )
    _checkpoint("Fitting trainer with CPU AMP fallback")
    trainer.fit()

    _checkpoint("Asserting steps completed and scaler disabled", global_step=trainer.state.global_step, is_enabled=trainer.scaler.is_enabled)
    assert trainer.state.global_step == 3
    assert trainer.scaler.is_enabled is False
    assert (tmp_path / "checkpoint_latest.pt").exists()


def test_trainer_subset_dataloader_resume_behavior(tmp_path, torch_device):
    """Cover H-03 issue verifying dataloader resume state behavior when using Subset."""
    _checkpoint("test_trainer_subset_dataloader_resume_behavior setup", device=torch_device)
    x = torch.randint(0, 11, (30, 5))
    y = torch.randint(0, 11, (30, 5))
    full_dataset = TensorDataset(x, y)
    subset = Subset(full_dataset, range(20))
    loader = DataLoader(subset, batch_size=4, shuffle=True)

    trainer = Trainer(
        model=TinyLM().to(torch_device),
        train_dataloader=loader,
        device=str(torch_device),
        max_epochs=1,
        max_train_steps=2,
        checkpoint_dir=str(tmp_path),
    )
    _checkpoint("Fitting trainer on subset dataloader")
    trainer.fit()
    ckpt = tmp_path / "checkpoint_latest.pt"
    assert ckpt.exists()

    resumed = Trainer(
        model=TinyLM().to(torch_device),
        train_dataloader=loader,
        device=str(torch_device),
        max_epochs=2,
        checkpoint_dir=str(tmp_path),
        resume_from=str(ckpt),
    )
    _checkpoint("Fitting resumed trainer on subset dataloader")
    resumed.fit()
    assert resumed.state.epoch >= 1
