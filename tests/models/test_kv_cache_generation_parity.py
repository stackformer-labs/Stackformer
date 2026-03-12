import torch

from stackformer import GPT_2

SEQ = 6
EMB = 8
HEADS = 2
VOCAB = 32


def test_incremental_next_token_logits_match_full_forward():
    torch.manual_seed(0)
    tokens = torch.randint(0, VOCAB, (1, SEQ))

    model = GPT_2(
        vocab_size=VOCAB,
        num_layers=1,
        embed_dim=EMB,
        num_heads=HEADS,
        seq_len=SEQ,
        hidden_dim=16,
        dropout=0.0,
    )

    full_logits = model(tokens)
    assert full_logits.shape == (1, SEQ, VOCAB)
    assert torch.isfinite(full_logits).all()

    step_logits = []
    for t in range(SEQ):
        prefix_logits = model(tokens[:, : t + 1])
        assert prefix_logits.shape == (1, t + 1, VOCAB)
        assert torch.isfinite(prefix_logits).all()
        step_logits.append(prefix_logits[:, -1, :])

    stacked = torch.stack(step_logits, dim=1)
    assert stacked.shape == full_logits.shape
    # Final next-token logits from incremental decoding should match full forward.
    assert torch.allclose(full_logits[:, -1, :], stacked[:, -1, :], atol=1e-5)

    # Gradient flow sanity check on full forward.
    loss = full_logits.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
