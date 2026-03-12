import pytest
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

BATCH, SEQ, EMB = 2, 6, 8
HEADS, KV_HEADS = 2, 1
MASKS = [
    dict(mask_type=["causal"]),
    dict(mask_type=["sliding_window"], window_size=2),
    dict(mask_type=["sliding_window", "global_mask"], window_size=2, global_index=[1]),
    dict(mask_type=["full"]),
]


def _assert_finite_and_grad(out: torch.Tensor, x: torch.Tensor) -> None:
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.abs(out).mean() < 100

    loss = out.square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("mask_kw", MASKS)
@pytest.mark.parametrize(
    "attn_ctor",
    [
        lambda kw: Self_Attention(EMB, dropout=0.0, **kw),
        lambda kw: Multi_Head_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0, **kw),
        lambda kw: Multi_Head_Attention_With_RoPE(embed_dim=EMB, num_heads=HEADS, dropout=0.0, **kw),
        lambda kw: Multi_query_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0, **kw),
        lambda kw: Multi_query_Attention_With_RoPE(embed_dim=EMB, num_heads=HEADS, dropout=0.0, **kw),
        lambda kw: Group_query_Attention(
            embed_dim=EMB,
            num_query_heads=HEADS,
            num_kv_heads=KV_HEADS,
            dropout=0.0,
            **kw,
        ),
        lambda kw: Group_query_Attention_With_RoPE(
            embed_dim=EMB,
            num_query_heads=HEADS,
            num_kv_heads=KV_HEADS,
            dropout=0.0,
            **kw,
        ),
    ],
)
def test_attention_variants_masks_and_gradients(mask_kw, attn_ctor):
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, EMB, requires_grad=True)
    out = attn_ctor(mask_kw)(x)
    _assert_finite_and_grad(out, x)


def test_cross_attention_shape_and_gradients():
    torch.manual_seed(0)
    q = torch.randn(BATCH, SEQ, EMB, requires_grad=True)
    ctx = torch.randn(BATCH, SEQ, EMB)
    layer = Cross_MultiHead_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0, mask_type=["causal"])

    out = layer(q, context=ctx)
    _assert_finite_and_grad(out, q)


@pytest.mark.parametrize("batch", [1, 2])
def test_kv_cache_multihead_multiple_decode_steps(batch):
    torch.manual_seed(0)
    cache = kv_cache_multihead(embed_dim=EMB, num_heads=HEADS, batch_size=2, kv_seq_len=SEQ, dropout=0.0)

    x0 = torch.randn(batch, 2, EMB)
    y0 = cache(x0, start_pos=0)
    assert y0.shape == x0.shape
    assert cache.cache_keys[:batch, :2].shape == (batch, 2, HEADS, EMB // HEADS)

    x1 = torch.randn(batch, 1, EMB)
    y1 = cache(x1, start_pos=2)
    assert y1.shape == x1.shape
    assert torch.isfinite(y1).all()
    assert torch.abs(y1).mean() < 100


def test_kv_cache_group_query_multiple_decode_steps():
    torch.manual_seed(0)
    cache = kv_cache_group_query(
        embed_dim=EMB,
        num_query_heads=HEADS,
        num_kv_heads=KV_HEADS,
        batch_size=BATCH,
        kv_seq_len=SEQ,
        dropout=0.0,
    )

    x0 = torch.randn(BATCH, 2, EMB)
    y0 = cache(x0, start_pos=0, rope=True)
    assert y0.shape == x0.shape
    assert cache.cache_keys[:, :2].shape == (BATCH, 2, KV_HEADS, EMB // HEADS)

    x1 = torch.randn(BATCH, 1, EMB)
    y1 = cache(x1, start_pos=2, rope=False)
    assert y1.shape == x1.shape
    assert torch.isfinite(y1).all()
    assert torch.abs(y1).mean() < 100


@pytest.mark.parametrize(
    "ctor,args,error",
    [
        (Multi_Head_Attention, dict(embed_dim=10, num_heads=3), AssertionError),
        (Group_query_Attention, dict(embed_dim=8, num_query_heads=3, num_kv_heads=2), AssertionError),
        (Self_Attention, dict(embed_dim=8, mask_type=["does_not_exist"]), ValueError),
    ],
)
def test_attention_invalid_configs_raise(ctor, args, error):
    with pytest.raises(error):
        layer = ctor(**args)
        if ctor is Self_Attention:
            layer(torch.randn(1, 4, 8))


def test_cross_attention_mismatched_batch_raises():
    layer = Cross_MultiHead_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0)
    q = torch.randn(2, 4, EMB)
    ctx = torch.randn(1, 4, EMB)
    with pytest.raises(RuntimeError):
        layer(q, context=ctx)


def test_cross_attention_accepts_explicit_t_s_mask():
    layer = Cross_MultiHead_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0)
    q = torch.randn(BATCH, 4, EMB, requires_grad=True)
    ctx = torch.randn(BATCH, 6, EMB)
    mask = torch.zeros(4, 6, dtype=torch.bool)
    out = layer(q, context=ctx, attn_mask=mask)
    _assert_finite_and_grad(out, q)


def test_cross_attention_invalid_mask_shape_raises():
    layer = Cross_MultiHead_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0)
    q = torch.randn(BATCH, 4, EMB)
    ctx = torch.randn(BATCH, 6, EMB)
    bad_mask = torch.zeros(4, 4, dtype=torch.bool)
    with pytest.raises(ValueError):
        layer(q, context=ctx, attn_mask=bad_mask)


def test_attention_constructors_allow_positional_dropout_safely():
    x = torch.randn(1, 3, EMB)
    # Historically some model blocks passed dropout positionally.
    layer = Multi_Head_Attention_With_RoPE(EMB, HEADS, 0.0)
    out = layer(x)
    assert out.shape == x.shape
