import torch

from stackformer.modules.Attention import (
    Cross_MultiHead_Attention,
    Group_query_Attention,
    Group_query_Attention_With_RoPE,
    Multi_Head_Attention,
    Multi_Head_Attention_With_RoPE,
    Multi_query_Attention,
    Multi_query_Attention_With_RoPE,
    Self_Attention,
    kv_cache_group_query,
    kv_cache_multihead,
)

BATCH, SEQ, EMB, HEADS, KV_HEADS = 2, 6, 8, 2, 1
MASKS = [
    dict(mask_type=["causal"]),
    dict(mask_type=["sliding_window"], window_size=2),
    dict(mask_type=["global_mask"], global_index=[0]),
    dict(mask_type=["dilated_causal"], dilation=2),
    dict(mask_type=["random_mask"], num_random=2),
    dict(mask_type=["sliding_window", "global_mask"], window_size=2, global_index=[1]),
    dict(mask_type=["full"]),
]
ATTENTIONS = [
    lambda kw: Self_Attention(EMB, dropout=0.0, **kw),
    lambda kw: Multi_Head_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0, **kw),
    lambda kw: Multi_Head_Attention_With_RoPE(embed_dim=EMB, num_heads=HEADS, dropout=0.0, **kw),
    lambda kw: Multi_query_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0, **kw),
    lambda kw: Multi_query_Attention_With_RoPE(embed_dim=EMB, num_heads=HEADS, dropout=0.0, **kw),
    lambda kw: Group_query_Attention(embed_dim=EMB, num_query_heads=HEADS, num_kv_heads=KV_HEADS, dropout=0.0, **kw),
    lambda kw: Group_query_Attention_With_RoPE(embed_dim=EMB, num_query_heads=HEADS, num_kv_heads=KV_HEADS, dropout=0.0, **kw),
]


def test_attention_variants_with_masks():
    for mask_kw in MASKS:
        for ctor in ATTENTIONS:
            x = torch.randn(BATCH, SEQ, EMB, requires_grad=True)
            out = ctor(mask_kw)(x)
            assert out.shape == x.shape
            assert not torch.isnan(out).any()
            out.mean().backward()


def test_cross_attention_and_kv_cache():
    q = torch.randn(BATCH, SEQ, EMB, requires_grad=True)
    ctx = torch.randn(BATCH, SEQ, EMB)
    out = Cross_MultiHead_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0, mask_type=["causal"])(q, context=ctx)
    assert out.shape == q.shape
    out.mean().backward()

    x = torch.randn(BATCH, SEQ, EMB)
    assert kv_cache_multihead(embed_dim=EMB, num_heads=HEADS, batch_size=BATCH, kv_seq_len=SEQ)(x, 0).shape == x.shape
    assert kv_cache_group_query(embed_dim=EMB, num_query_heads=HEADS, num_kv_heads=KV_HEADS, batch_size=BATCH, kv_seq_len=SEQ)(x, 0, True).shape == x.shape
