import torch

from stackformer import GPT_2

BATCH = 2
SEQ = 6
EMB = 8
HEADS = 2
VOCAB = 32


def test_full_stackformer_pipeline_forward_backward_and_generation():
    torch.manual_seed(0)

    tokens = torch.randint(0, VOCAB, (BATCH, SEQ))
    model = GPT_2(
        vocab_size=VOCAB,
        num_layers=1,
        embed_dim=EMB,
        num_heads=HEADS,
        seq_len=SEQ,
        hidden_dim=16,
        dropout=0.0,
    )

    logits = model(tokens)
    assert logits.shape == (BATCH, SEQ, VOCAB)
    assert torch.isfinite(logits).all()

    loss = logits.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

    generated = model.generate(tokens[0], max_context_len=SEQ, max_new_tokens=2)
    assert generated.shape == (1, SEQ + 2)
    assert torch.isfinite(generated.float()).all()
