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

# Small safe sizes
BATCH = 2
SEQ = 6
EMB = 8
HEADS = 2
KV_HEADS = 1
WINDOW = 2

# Mask configurations
MASKS = [
    dict(mask_type=["causal"]),
    dict(mask_type=["sliding_window"], window_size=2),
    dict(mask_type=["global_mask"], global_index=[0]),
    dict(mask_type=["dilated_causal"], dilation=2),
    dict(mask_type=["random_mask"], num_random=2),
    dict(mask_type=["sliding_window", "global_mask"], window_size=2, global_index=[1]),
    dict(mask_type=["full"]),
]

# Attention constructors
ATTENTIONS = [
    lambda kw: Self_Attention(EMB, dropout=0.0, **kw),
    lambda kw: Multi_Head_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0, **kw),
    lambda kw: Multi_Head_Attention_With_RoPE(embed_dim=EMB, num_heads=HEADS, dropout=0.0, **kw),
    lambda kw: Multi_query_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0, **kw),
    lambda kw: Multi_query_Attention_With_RoPE(embed_dim=EMB, num_heads=HEADS, dropout=0.0, **kw),
    lambda kw: Group_query_Attention(embed_dim=EMB, num_query_heads=HEADS, num_kv_heads=KV_HEADS, dropout=0.0, **kw),
    lambda kw: Group_query_Attention_With_RoPE(embed_dim=EMB, num_query_heads=HEADS, num_kv_heads=KV_HEADS, dropout=0.0, **kw),
]

# Core mask × attention test
@pytest.mark.parametrize("mask_kw", MASKS)
@pytest.mark.parametrize("ctor", ATTENTIONS)
def test_attention_variants_with_masks(mask_kw, ctor):
    x = torch.randn(BATCH, SEQ, EMB, requires_grad=True)

    attn = ctor(mask_kw)
    out = attn(x)

    assert out.shape == x.shape
    assert not torch.isnan(out).any()

    out.mean().backward()

# Cross attention (special case)
@pytest.mark.parametrize("mask_kw", MASKS)
def test_cross_attention(mask_kw):
    q = torch.randn(BATCH, SEQ, EMB, requires_grad=True)
    ctx = torch.randn(BATCH, SEQ, EMB)

    cross = Cross_MultiHead_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0, **mask_kw)
    out = cross(q, context=ctx)

    assert out.shape == q.shape
    assert not torch.isnan(out).any()

    out.mean().backward()

# Local attention (unchanged)
def test_local_attention():
    x = torch.randn(BATCH, SEQ, EMB)
    local = Local_Attention(embed_dim=EMB, window_size=WINDOW, num_heads=HEADS, dropout=0.0)
    out = local(x)
    assert out.shape == x.shape

# KV cache tests (unchanged)
def test_kv_cache_multihead():
    x = torch.randn(BATCH, SEQ, EMB)
    kv = kv_cache_multihead(embed_dim=EMB, num_heads=HEADS, batch_size=BATCH, kv_seq_len=SEQ)
    out = kv(x, 0)
    assert out.shape == x.shape


def test_kv_cache_group_query():
    x = torch.randn(BATCH, SEQ, EMB)
    kv = kv_cache_group_query(embed_dim=EMB, num_query_heads=HEADS, num_kv_heads=KV_HEADS, batch_size=BATCH, kv_seq_len=SEQ)
    out = kv(x, 0, True)
    assert out.shape == x.shape
