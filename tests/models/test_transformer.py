import pytest
import torch

from stackformer.models.Transformer import transformer
from stackformer.modules.Attention import Multi_Head_Attention
from stackformer.modules.Feed_forward import FF_ReLU
from stackformer.modules.position_embedding import SinusoidalPositionalEmbedding


@pytest.mark.parametrize("batch,seq,embed_dim,heads", [(1, 4, 8, 1), (2, 6, 16, 4)])
def test_transformer_forward_shape_finite_and_backward(batch, seq, embed_dim, heads):
    torch.manual_seed(0)
    vocab_size = 32
    x = torch.randint(0, vocab_size, (batch, seq))

    model = transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=heads,
        dropout=0.0,
        hidden_dim=embed_dim * 2,
        encoder_layers=1,
        decoder_layers=1,
        seq_len=seq,
    )
    out = model(x, x)
    assert out.shape == (batch, seq, vocab_size)
    assert torch.isfinite(out).all()
    assert torch.abs(out).mean() < 100

    out.mean().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_module_interoperability_embedding_pos_attention_ff_pipeline():
    torch.manual_seed(0)
    batch, seq, vocab_size, embed_dim = 2, 6, 32, 8
    token_embedding = torch.nn.Embedding(vocab_size, embed_dim)
    pos_embedding = SinusoidalPositionalEmbedding(seq_len=seq, embed_dim=embed_dim)
    attention = Multi_Head_Attention(embed_dim=embed_dim, num_heads=2, mask_type=["full"], dropout=0.0)
    ff = FF_ReLU(embed_dim=embed_dim, hidden_dim=16, dropout=0.0)

    ids = torch.randint(0, vocab_size, (batch, seq))
    x = token_embedding(ids)
    x = x + pos_embedding(x)
    x = attention(x)
    x = ff(x)

    assert x.shape == (batch, seq, embed_dim)
    assert torch.isfinite(x).all()

    x.mean().backward()
    assert token_embedding.weight.grad is not None


def test_transformer_invalid_head_setup_raises():
    with pytest.raises(AssertionError):
        transformer(
            vocab_size=20,
            embed_dim=10,
            num_heads=3,
            dropout=0.0,
            hidden_dim=16,
            encoder_layers=1,
            decoder_layers=1,
            seq_len=4,
        )
