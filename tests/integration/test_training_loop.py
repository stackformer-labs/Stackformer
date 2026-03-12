import torch

from stackformer import GPT_1


def test_minimal_training_loop_runs_and_updates_weights():
    torch.manual_seed(0)
    model = GPT_1(vocab_size=32, num_layers=1, embed_dim=8, num_heads=2, seq_len=6, dropout=0.0, hidden_dim=16)
    x = torch.randint(0, 32, (2, 6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    before = model.lm_head.weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        assert torch.isfinite(out).all()
        loss = out.mean()
        loss.backward()
        optimizer.step()

    assert not torch.allclose(before, model.lm_head.weight.detach())
