import torch

torch.set_default_device("cpu")

from stackformer.modules.Attention import Multi_Head_Attention, Self_Attention
from stackformer.modules.Feed_forward import FF_GELU
from stackformer.modules.Normalization import LayerNormalization, RMSNormalization
from stackformer.modules.position_embedding import AbsolutePositionEmbedding, SinusoidalPositionalEmbedding


def test_core_module_shapes() -> None:
    x = torch.randn(2, 8, 16)

    sa = Self_Attention(embed_dim=16, num_heads=4, dropout=0.0)
    mha = Multi_Head_Attention(embed_dim=16, num_heads=4, dropout=0.0)
    ff = FF_GELU(embed_dim=16, hidden_dim=32, dropout=0.0)
    ln = LayerNormalization(embed_dim=16)
    rn = RMSNormalization(embed_dim=16)

    assert sa(x).shape == x.shape
    assert mha(x).shape == x.shape
    assert ff(x).shape == x.shape
    assert ln(x).shape == x.shape
    assert rn(x).shape == x.shape


def test_position_embeddings_shape() -> None:
    token_ids = torch.randint(0, 10, (2, 8))

    abs_pos = AbsolutePositionEmbedding(seq_len=16, embed_dim=16)
    sin_pos = SinusoidalPositionalEmbedding(seq_len=16, embed_dim=16)

    assert abs_pos(token_ids).shape == (2, 8, 16)
    assert sin_pos(token_ids).shape == (2, 8, 16)
