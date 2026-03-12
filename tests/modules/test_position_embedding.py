import pytest
import torch

from stackformer.modules.position_embedding import AbsolutePositionEmbedding, RoPE, SinusoidalPositionalEmbedding


@pytest.mark.parametrize("batch,seq", [(1, 4), (2, 6)])
def test_absolute_position_embedding_shapes_and_stability(batch, seq):
    torch.manual_seed(0)
    layer = AbsolutePositionEmbedding(seq_len=8, embed_dim=8)
    x = torch.randint(0, 16, (batch, seq))

    out = layer(x)
    assert out.shape == (batch, seq, 8)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("batch,seq", [(1, 4), (2, 6)])
def test_sinusoidal_position_embedding_shapes_and_stability(batch, seq):
    torch.manual_seed(0)
    layer = SinusoidalPositionalEmbedding(seq_len=8, embed_dim=8)
    x = torch.randn(batch, seq, 8)

    out = layer(x)
    assert out.shape == (batch, seq, 8)
    assert torch.isfinite(out).all()


def test_rope_compatibility_with_attention_shape_and_stability():
    torch.manual_seed(0)
    rope = RoPE(head_dim=4, seq_len=8)
    x = torch.randn(2, 2, 6, 4)

    out = rope(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_rope_requires_even_head_dim():
    with pytest.raises(AssertionError):
        RoPE(head_dim=3, seq_len=6)


def test_absolute_position_embedding_fails_when_seq_is_too_long():
    layer = AbsolutePositionEmbedding(seq_len=4, embed_dim=8)
    with pytest.raises(IndexError):
        layer(torch.randint(0, 10, (2, 5)))


def test_rope_fails_for_odd_input_last_dim():
    rope = RoPE(head_dim=4, seq_len=8)
    with pytest.raises(AssertionError):
        rope(torch.randn(1, 2, 4, 3))
