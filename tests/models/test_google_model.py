import torch

from stackformer import gemma_1_2b
from stackformer.models.Google import gemma_1_7b


def test_google_models_forward():
    torch.manual_seed(0)
    vocab_size, seq_len, batch_size = 32, 6, 2
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    for model_cls in (gemma_1_2b, gemma_1_7b):
        model = model_cls(
            vocab_size=vocab_size,
            num_layers=1,
            embed_dim=8,
            num_heads=2,
            seq_len=seq_len,
            dropout=0.0,
            hidden_dim=16,
        )
        assert model(x).shape == (batch_size, seq_len, vocab_size)
