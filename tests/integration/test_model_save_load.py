import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stackformer.engine import Trainer


class TinyTokenModel(nn.Module):
    def __init__(self, vocab_size=13, hidden=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.proj(self.embedding(x))


def test_training_save_load_and_resume(tmp_path):
    x = torch.randint(0, 13, (24, 6))
    y = x.clone()
    loader = DataLoader(TensorDataset(x, y), batch_size=6)

    trainer = Trainer(model=TinyTokenModel(), train_dataloader=loader, device="cpu", max_epochs=2, checkpoint_dir=str(tmp_path))
    trainer.fit()
    ckpt = tmp_path / "checkpoint_latest.pt"
    assert ckpt.exists()

    reloaded = TinyTokenModel()
    resumed = Trainer(
        model=reloaded,
        train_dataloader=loader,
        device="cpu",
        max_epochs=3,
        checkpoint_dir=str(tmp_path),
        resume_from=str(ckpt),
    )
    resumed.fit()

    batch = next(iter(loader))[0]
    out = reloaded(batch)
    assert out.shape == (6, 6, 13)
    assert resumed.state.epoch >= 3
