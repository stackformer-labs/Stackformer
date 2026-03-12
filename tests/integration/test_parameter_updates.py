import torch

from stackformer import GPT_1

BATCH = 2
SEQ = 6
EMB = 8
HEADS = 2
VOCAB = 32


def test_parameter_update_occurs_after_optimizer_step():
    torch.manual_seed(0)

    model = GPT_1(
        vocab_size=VOCAB,
        num_layers=1,
        embed_dim=EMB,
        num_heads=HEADS,
        seq_len=SEQ,
        hidden_dim=16,
        dropout=0.0,
    )
    x = torch.randint(0, VOCAB, (BATCH, SEQ))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    old_weight = model.embedding.weight.detach().clone()

    logits = model(x)
    assert torch.isfinite(logits).all()
    loss = logits.mean()
    loss.backward()
    optimizer.step()

    assert not torch.allclose(old_weight, model.embedding.weight.detach())
