import torch

from stackformer.modules.position_embedding import (
    AbsolutePositionEmbedding,
    SinusoidalPositionalEmbedding,
    RoPE,
)

BATCH = 2
SEQ = 6
EMB = 8          # must be even
HEADS = 2
HEAD_DIM = EMB // HEADS


def test_absolute_position_embedding_shape():
    x = torch.randint(0, 10, (BATCH, SEQ))

    pos = AbsolutePositionEmbedding(seq_len=SEQ, embed_dim=EMB)
    out = pos(x)

    assert out.shape == (BATCH, SEQ, EMB)
    assert torch.isfinite(out).all()


def test_sinusoidal_position_embedding_shape():
    x = torch.randn(BATCH, SEQ, EMB)

    pos = SinusoidalPositionalEmbedding(seq_len=SEQ, embed_dim=EMB)
    out = pos(x)

    assert out.shape == (BATCH, SEQ, EMB)
    assert torch.isfinite(out).all()


def test_rope_forward():
    # RoPE expects (B, H, T, D)
    x = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM)

    rope = RoPE(head_dim=HEAD_DIM, seq_len=SEQ)
    out = rope(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_rope_even_head_dim_assert():
    # odd head_dim should fail
    bad_dim = 3
    try:
        RoPE(head_dim=bad_dim, seq_len=SEQ)
        assert False, "RoPE should assert on odd head_dim"
    except AssertionError:
        pass