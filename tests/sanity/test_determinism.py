import torch

from stackformer import GPT_1
from stackformer.modules.Attention import Multi_Head_Attention
from stackformer.modules.position_embedding import AbsolutePositionEmbedding, RoPE, SinusoidalPositionalEmbedding


def test_attention_determinism_fixed_seed():
    x = torch.randn(2, 6, 8)

    torch.manual_seed(0)
    m1 = Multi_Head_Attention(embed_dim=8, num_heads=2, dropout=0.0)
    y1 = m1(x)

    torch.manual_seed(0)
    m2 = Multi_Head_Attention(embed_dim=8, num_heads=2, dropout=0.0)
    y2 = m2(x)

    assert torch.allclose(y1, y2)


def test_position_embeddings_determinism_fixed_seed():
    ids = torch.randint(0, 8, (2, 6))

    torch.manual_seed(0)
    abs1 = AbsolutePositionEmbedding(seq_len=6, embed_dim=8)
    a1 = abs1(ids)

    torch.manual_seed(0)
    abs2 = AbsolutePositionEmbedding(seq_len=6, embed_dim=8)
    a2 = abs2(ids)

    s = SinusoidalPositionalEmbedding(seq_len=6, embed_dim=8)
    r = RoPE(head_dim=4, seq_len=6)
    x = torch.randn(2, 2, 6, 4)

    assert torch.allclose(a1, a2)
    assert torch.allclose(s(ids), s(ids))
    assert torch.allclose(r(x), r(x))


def test_gpt_determinism_fixed_seed():
    x = torch.randint(0, 32, (2, 6))

    torch.manual_seed(0)
    m1 = GPT_1(vocab_size=32, num_layers=1, embed_dim=8, num_heads=2, seq_len=6, dropout=0.0, hidden_dim=16)
    y1 = m1(x)

    torch.manual_seed(0)
    m2 = GPT_1(vocab_size=32, num_layers=1, embed_dim=8, num_heads=2, seq_len=6, dropout=0.0, hidden_dim=16)
    y2 = m2(x)

    assert torch.allclose(y1, y2)
