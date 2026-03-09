import torch

from stackformer import llama_1
from stackformer.models.Meta import llama_2


def test_meta_models_forward():
    torch.manual_seed(0)
    vocab_size, seq_len, batch_size = 32, 6, 2
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    l1 = llama_1(
        vocab_size=vocab_size,
        num_layers=1,
        embed_dim=8,
        num_heads=2,
        seq_len=seq_len,
        dropout=0.0,
        hidden_dim=16,
    )
    assert l1(x).shape == (batch_size, seq_len, vocab_size)

    l2 = llama_2(
        vocab_size=vocab_size,
        num_layers=1,
        embed_dim=8,
        num_query_heads=2,
        num_kv_heads=1,
        kv_seq_len=seq_len,
        batch_size=batch_size,
        hidden_dim=16,
        dropout=0.0,
    )
    assert l2(x).shape == (batch_size, seq_len, vocab_size)
