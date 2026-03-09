import torch
from torch import nn

from stackformer.engine import CheckpointManager


def test_checkpoint_save_and_load(tmp_path):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    manager = CheckpointManager(output_dir=str(tmp_path), device="cpu")

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    manager.save({"model": model, "optimizer": optimizer, "epoch": 2, "global_step": 7}, name="unit")

    restored = nn.Linear(3, 2)
    restored_opt = torch.optim.Adam(restored.parameters(), lr=1e-3)
    meta = manager.load(str(tmp_path / "checkpoint_unit.pt"), {"model": restored, "optimizer": restored_opt})

    assert meta["epoch"] == 2
    assert meta["global_step"] == 7
    for p1, p2 in zip(model.parameters(), restored.parameters()):
        assert torch.allclose(p1, p2)


def test_checkpoint_save_skips_non_main_process(tmp_path, monkeypatch):
    model = nn.Linear(2, 2)
    manager = CheckpointManager(output_dir=str(tmp_path), device="cpu")

    monkeypatch.setattr("stackformer.engine.checkpoint.is_main_process", lambda: False)
    result = manager.save({"model": model}, name="nope")

    assert result is None
    assert not (tmp_path / "checkpoint_nope.pt").exists()
