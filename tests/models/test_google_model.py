import torch

from stackformer import gemma_1_2b
from stackformer.models.Google import gemma_1_7b


@torch.no_grad()
def _check_model_forward(model_cls):
    vocab_size, batch, seq_len = 32, 2, 6
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
    logits = model(x)
    assert logits.shape == (batch, seq_len, vocab_size)
    assert torch.isfinite(logits).all()
    assert torch.abs(logits).mean() < 100


def test_google_models_forward_and_generate():
    torch.manual_seed(0)
    for model_cls in (gemma_1_2b, gemma_1_7b):
        _check_model_forward(model_cls)

    x = torch.randint(0, 32, (1, 6))
    model = gemma_1_2b(vocab_size=32, num_layers=1, embed_dim=8, num_heads=2, seq_len=6, dropout=0.0, hidden_dim=16)
    generated = model.generate(x[0], max_context_len=6, max_new_tokens=2)
    assert generated.shape == (1, 8)
