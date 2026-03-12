import pytest
import torch

from stackformer import GPT_1, GPT_2


@pytest.mark.parametrize("model_cls", [GPT_1, GPT_2])
def test_save_load_after_training_keeps_predictions(tmp_path, model_cls):
    torch.manual_seed(0)
    x = torch.randint(0, 32, (2, 6))

    model = model_cls(vocab_size=32, num_layers=1, embed_dim=8, num_heads=2, seq_len=6, dropout=0.0, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = logits.mean()
        loss.backward()
        optimizer.step()

    model.eval()
    expected = model(x)

    path = tmp_path / "trained.pt"
    torch.save(model.state_dict(), path)

    reloaded = model_cls(vocab_size=32, num_layers=1, embed_dim=8, num_heads=2, seq_len=6, dropout=0.0, hidden_dim=16)
    reloaded.load_state_dict(torch.load(path, map_location="cpu"))
    reloaded.eval()

    got = reloaded(x)
    assert torch.isfinite(got).all()
    assert torch.allclose(expected, got)
