import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stackformer.engine import CheckpointManager, Trainer


class TinyLM(nn.Module):
    def __init__(self, vocab_size=11, hidden=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.proj(self.embedding(x))


def test_checkpoint_manager_save_and_load(tmp_path):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    manager = CheckpointManager(output_dir=str(tmp_path), device="cpu")
    manager.save({"model": model, "optimizer": optimizer, "epoch": 2, "global_step": 7}, name="unit")

    restored = nn.Linear(3, 2)
    restored_opt = torch.optim.Adam(restored.parameters(), lr=1e-3)
    meta = manager.load(str(tmp_path / "checkpoint_unit.pt"), {"model": restored, "optimizer": restored_opt})

    assert meta["epoch"] == 2
    assert meta["global_step"] == 7


def test_resume_training_from_checkpoint(tmp_path):
    x = torch.randint(0, 11, (12, 4))
    y = torch.randint(0, 11, (12, 4))
    loader = DataLoader(TensorDataset(x, y), batch_size=3)

    trainer = Trainer(model=TinyLM(), train_dataloader=loader, device="cpu", max_epochs=1, checkpoint_dir=str(tmp_path))
    trainer.fit()
    ckpt_path = tmp_path / "checkpoint_latest.pt"
    assert ckpt_path.exists()

    resumed = Trainer(
        model=TinyLM(),
        train_dataloader=loader,
        device="cpu",
        max_epochs=2,
        checkpoint_dir=str(tmp_path),
        resume_from=str(ckpt_path),
    )
    resumed.fit()
    assert resumed.state.epoch >= 2
