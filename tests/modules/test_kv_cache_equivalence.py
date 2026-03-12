import torch

from stackformer.modules.Attention import (
    Group_query_Attention,
    Multi_Head_Attention,
    kv_cache_group_query,
    kv_cache_multihead,
)


def _copy_mha_weights(src, dst):
    with torch.no_grad():
        dst.q_proj.weight.copy_(src.q_proj.weight)
        dst.k_proj.weight.copy_(src.k_proj.weight)
        dst.v_proj.weight.copy_(src.v_proj.weight)
        dst.out_proj.weight.copy_(src.out_proj.weight)
        if src.q_proj.bias is not None and dst.q_proj.bias is not None:
            dst.q_proj.bias.copy_(src.q_proj.bias)
            dst.k_proj.bias.copy_(src.k_proj.bias)
            dst.v_proj.bias.copy_(src.v_proj.bias)
        if src.out_proj.bias is not None and dst.out_proj.bias is not None:
            dst.out_proj.bias.copy_(src.out_proj.bias)


def test_kv_cache_multihead_matches_full_attention_for_last_token():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 8)

    full = Multi_Head_Attention(embed_dim=8, num_heads=2, dropout=0.0, mask_type=["causal"])
    cached = kv_cache_multihead(embed_dim=8, num_heads=2, batch_size=2, kv_seq_len=6, dropout=0.0)
    _copy_mha_weights(full, cached)

    full_out = full(x)

    step_out = None
    for t in range(x.size(1)):
        step_out = cached(x[:, t : t + 1], start_pos=t, rope=False)

    assert step_out is not None
    assert step_out.shape == (2, 1, 8)
    assert torch.allclose(step_out.squeeze(1), full_out[:, -1, :], atol=1e-5)


def test_kv_cache_group_query_matches_full_attention_for_last_token():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 8)

    full = Group_query_Attention(embed_dim=8, num_query_heads=2, num_kv_heads=1, dropout=0.0, mask_type=["causal"])
    cached = kv_cache_group_query(embed_dim=8, num_query_heads=2, num_kv_heads=1, batch_size=2, kv_seq_len=6, dropout=0.0)
    _copy_mha_weights(full, cached)

    full_out = full(x)

    step_out = None
    for t in range(x.size(1)):
        step_out = cached(x[:, t : t + 1], start_pos=t, rope=False)

    assert step_out is not None
    assert step_out.shape == (2, 1, 8)
    assert torch.allclose(step_out.squeeze(1), full_out[:, -1, :], atol=1e-5)
