import pytest
import torch

from stackformer import GPT_1, GPT_2


@pytest.mark.parametrize("model_cls", [GPT_1, GPT_2])
@pytest.mark.parametrize("batch,seq,embed_dim,heads", [(1, 4, 8, 1), (2, 6, 16, 4)])
def test_gpt_forward_shapes_gradients_and_finiteness(model_cls, batch, seq, embed_dim, heads):
    torch.manual_seed(0)
    vocab_size = 32
    x = torch.randint(0, vocab_size, (batch, seq))

    model = model_cls(
        vocab_size=vocab_size,
        num_layers=1,
        embed_dim=embed_dim,
        num_heads=heads,
        seq_len=seq,
        dropout=0.0,
        hidden_dim=embed_dim * 2,
    )

    logits = model(x)
    assert logits.shape == (batch, seq, vocab_size)
    assert torch.isfinite(logits).all()
    assert torch.abs(logits).mean() < 100

    loss = logits.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("model_cls", [GPT_1, GPT_2])
def test_gpt_generate_handles_single_and_batch_inputs(model_cls):
    torch.manual_seed(0)
    vocab_size, seq_len = 32, 6
    x = torch.randint(0, vocab_size, (2, seq_len))

    model = model_cls(
        vocab_size=vocab_size,
        num_layers=1,
        embed_dim=8,
        num_heads=2,
        seq_len=seq_len,
        dropout=0.0,
        hidden_dim=16,
    )

    single = model.generate(x[0], max_context_len=seq_len, max_new_tokens=2)
    batch = model.generate(x, max_context_len=seq_len, max_new_tokens=3)
    clipped = model.generate(x, max_context_len=3, max_new_tokens=2)

    assert single.shape == (1, seq_len + 2)
    assert batch.shape == (2, seq_len + 3)
    assert clipped.shape == (2, seq_len + 2)


@pytest.mark.parametrize("model_cls", [GPT_1, GPT_2])
def test_gpt_minimal_training_loop_updates_parameters(model_cls):
    torch.manual_seed(0)
    vocab_size, batch, seq_len = 24, 2, 5
    x = torch.randint(0, vocab_size, (batch, seq_len))

    model = model_cls(
        vocab_size=vocab_size,
        num_layers=1,
        embed_dim=8,
        num_heads=2,
        seq_len=seq_len,
        dropout=0.0,
        hidden_dim=16,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    before = model.lm_head.weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = logits.mean()
        loss.backward()
        optimizer.step()

    assert not torch.allclose(before, model.lm_head.weight.detach())


def test_gpt_invalid_head_configuration_raises():
    with pytest.raises(AssertionError):
        GPT_1(
            vocab_size=20,
            num_layers=1,
            embed_dim=10,
            num_heads=3,
            seq_len=6,
            dropout=0.0,
            hidden_dim=20,
        )
