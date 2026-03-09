import torch

from stackformer.models.Transformer import transformer


def test_transformer_forward():
    vocab_size, seq_len, batch_size = 32, 6, 2
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    seq2seq = transformer(
        vocab_size=vocab_size,
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
        hidden_dim=16,
        encoder_layers=1,
        decoder_layers=1,
        seq_len=seq_len,
    )
    y = seq2seq(x, x)
    assert y.shape == (batch_size, seq_len, vocab_size)
