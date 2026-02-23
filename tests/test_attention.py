import torch

from stackformer.modules.Attention import (
    Self_Attention,
    Multi_Head_Attention,
    Multi_Head_Attention_With_RoPE,
    Cross_MultiHead_Attention,
    Multi_query_Attention,
    Multi_query_Attention_With_RoPE,
    Group_query_Attention,
    Group_query_Attention_With_RoPE,
    Local_Attention,
    kv_cache_multihead,
    kv_cache_group_query,
)


# Use tiny safe sizes
BATCH = 2
SEQ = 6
EMB = 8          # must allow even head_dim
HEADS = 2        # 8 / 2 = 4 (even) ✅
KV_HEADS = 1     # for GQA
WINDOW = 2


def test_self_attention():
    x = torch.randn(BATCH, SEQ, EMB)
    sa = Self_Attention(EMB, dropout=0.0)
    out = sa(x)
    assert out.shape == x.shape


def test_multi_head_attention():
    x = torch.randn(BATCH, SEQ, EMB)
    mha = Multi_Head_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0)
    out = mha(x)
    assert out.shape == x.shape


def test_multi_head_attention_with_rope():
    x = torch.randn(BATCH, SEQ, EMB)
    mha = Multi_Head_Attention_With_RoPE(embed_dim=EMB, num_heads=HEADS, dropout=0.0)
    out = mha(x)
    assert out.shape == x.shape


def test_cross_multi_head_attention():
    q = torch.randn(BATCH, SEQ, EMB)
    context = torch.randn(BATCH, SEQ, EMB)

    cross = Cross_MultiHead_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0)
    out = cross(q, context=context)
    assert out.shape == q.shape


def test_multi_query_attention():
    x = torch.randn(BATCH, SEQ, EMB)
    mqa = Multi_query_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0)
    out = mqa(x)
    assert out.shape == x.shape


def test_multi_query_attention_with_rope():
    x = torch.randn(BATCH, SEQ, EMB)
    mqa = Multi_query_Attention_With_RoPE(embed_dim=EMB, num_heads=HEADS, dropout=0.0)
    out = mqa(x)
    assert out.shape == x.shape


def test_group_query_attention():
    x = torch.randn(BATCH, SEQ, EMB)
    gqa = Group_query_Attention(
        embed_dim=EMB,
        num_query_heads=HEADS,
        num_kv_heads=KV_HEADS,
        dropout=0.0,
    )
    out = gqa(x)
    assert out.shape == x.shape


def test_group_query_attention_with_rope():
    x = torch.randn(BATCH, SEQ, EMB)
    gqa = Group_query_Attention_With_RoPE(
        embed_dim=EMB,
        num_query_heads=HEADS,
        num_kv_heads=KV_HEADS,
        dropout=0.0,
    )
    out = gqa(x)
    assert out.shape == x.shape


def test_local_attention():
    x = torch.randn(BATCH, SEQ, EMB)
    local = Local_Attention(
        embed_dim=EMB,
        window_size=WINDOW,
        num_heads=HEADS,
        dropout=0.0,
    )
    out = local(x)
    assert out.shape == x.shape


def test_kv_cache_multihead():
    x = torch.randn(BATCH, SEQ, EMB)
    start_pos = 2

    kv = kv_cache_multihead(
        embed_dim=EMB,
        num_heads=HEADS,
        batch_size=BATCH,
        kv_seq_len=SEQ,
    )

    out = kv(x, start_pos)
    assert out.shape == x.shape


def test_kv_cache_group_query():
    x = torch.randn(BATCH, SEQ, EMB)
    start_pos = 2

    kv_gqa = kv_cache_group_query(
        embed_dim=EMB,
        num_query_heads=HEADS,
        num_kv_heads=KV_HEADS,
        batch_size=BATCH,
        kv_seq_len=SEQ,
    )

    out = kv_gqa(x, start_pos, True)
    assert out.shape == x.shape
