import torch

from stackformer.modules.Attention import Multi_Head_Attention
from stackformer.modules.Feed_forward import FF_GELU
from stackformer.modules.position_embedding import AbsolutePositionEmbedding


def test_embedding_position_attention_ff_pipeline_end_to_end():
    torch.manual_seed(0)
    tokens = torch.randint(0, 32, (2, 6))

    token_emb = torch.nn.Embedding(32, 8)
    pos_emb = AbsolutePositionEmbedding(seq_len=6, embed_dim=8)
    attn = Multi_Head_Attention(embed_dim=8, num_heads=2, dropout=0.0)
    ff = FF_GELU(embed_dim=8, hidden_dim=16, dropout=0.0)

    x = token_emb(tokens)
    x = x + pos_emb(tokens)
    x = attn(x)
    x = ff(x)

    assert x.shape == (2, 6, 8)
    assert torch.isfinite(x).all()
    assert torch.abs(x).mean() < 100

    x.mean().backward()
    assert token_emb.weight.grad is not None
