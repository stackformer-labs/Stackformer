import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tests._test_utils import _checkpoint

from stackformer.engine import CheckpointManager, Trainer


class TinyLM(nn.Module):
    def __init__(self, vocab_size=11, hidden=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.proj(self.embedding(x))


def test_checkpoint_manager_save_and_load(tmp_path, torch_device):
    _checkpoint("test_checkpoint_manager_save_and_load setup", device=torch_device)
    model = nn.Linear(3, 2, device=torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    manager = CheckpointManager(output_dir=str(tmp_path), device=str(torch_device))

    _checkpoint("Saving checkpoint via CheckpointManager")
    manager.save({"model": model, "optimizer": optimizer, "epoch": 2, "global_step": 7}, name="unit")

    restored = nn.Linear(3, 2, device=torch_device)
    restored_opt = torch.optim.Adam(restored.parameters(), lr=1e-3)

    _checkpoint("Loading checkpoint via CheckpointManager")
    meta = manager.load(str(tmp_path / "checkpoint_unit.pt"), {"model": restored, "optimizer": restored_opt})

    _checkpoint("Asserting restored metadata fields")
    assert meta["epoch"] == 2
    assert meta["global_step"] == 7


def test_resume_training_from_checkpoint(tmp_path, torch_device):
    _checkpoint("test_resume_training_from_checkpoint setup", device=torch_device)
    x = torch.randint(0, 11, (12, 4))
    y = torch.randint(0, 11, (12, 4))
    loader = DataLoader(TensorDataset(x, y), batch_size=3)

    _checkpoint("Fitting epoch 1 trainer")
    trainer = Trainer(model=TinyLM().to(torch_device), train_dataloader=loader, device=str(torch_device), max_epochs=1, checkpoint_dir=str(tmp_path))
    trainer.fit()
    ckpt_path = tmp_path / "checkpoint_latest.pt"
    assert ckpt_path.exists()

    _checkpoint("Resuming trainer for epoch 2")
    resumed = Trainer(
        model=TinyLM().to(torch_device),
        train_dataloader=loader,
        device=str(torch_device),
        max_epochs=2,
        checkpoint_dir=str(tmp_path),
        resume_from=str(ckpt_path),
    )
    resumed.fit()

    _checkpoint("Asserting resumed trainer reached expected epoch", epoch=resumed.state.epoch)
    assert resumed.state.epoch >= 2
