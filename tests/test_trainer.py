import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stackformer.engine import Trainer


class TinyLM(nn.Module):
    def __init__(self, vocab_size=11, hidden=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.proj(self.embedding(x))


def test_trainer_loop_executes_with_amp_flag_on_cpu(tmp_path):
    x = torch.randint(0, 11, (20, 5))
    y = torch.randint(0, 11, (20, 5))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    trainer = Trainer(
        model=TinyLM(),
        train_dataloader=loader,
        val_dataloader=loader,
        device="cpu",
        use_amp=True,
        max_epochs=1,
        max_train_steps=3,
        max_eval_steps=1,
        checkpoint_dir=str(tmp_path),
    )
    trainer.fit()

    assert trainer.state.global_step == 3
    assert trainer.scaler.is_enabled is False
    assert (tmp_path / "checkpoint_latest.pt").exists()


def test_trainer_ddp_flag_is_safe_in_single_process_cpu(tmp_path, monkeypatch):
    x = torch.randint(0, 11, (8, 5))
    y = torch.randint(0, 11, (8, 5))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    trainer = Trainer(
        model=TinyLM(),
        train_dataloader=loader,
        device="cpu",
        use_amp=True,
        use_ddp=True,
        ddp_backend="gloo",
        max_epochs=1,
        max_train_steps=2,
        checkpoint_dir=str(tmp_path),
    )
    trainer.fit()

    assert trainer.state.global_step == 2
    assert (tmp_path / "checkpoint_latest.pt").exists()
