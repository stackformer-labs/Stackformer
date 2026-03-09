import pytest
import torch

from stackformer.modules.position_embedding import AbsolutePositionEmbedding, RoPE, SinusoidalPositionalEmbedding


def test_position_embeddings_and_rope_shapes():
    assert AbsolutePositionEmbedding(seq_len=6, embed_dim=8)(torch.randint(0, 10, (2, 6))).shape == (2, 6, 8)
    assert SinusoidalPositionalEmbedding(seq_len=6, embed_dim=8)(torch.randn(2, 6, 8)).shape == (2, 6, 8)
    rope_out = RoPE(head_dim=4, seq_len=6)(torch.randn(2, 2, 6, 4))
    assert rope_out.shape == (2, 2, 6, 4)


def test_rope_requires_even_head_dim():
    with pytest.raises(AssertionError):
        RoPE(head_dim=3, seq_len=6)
