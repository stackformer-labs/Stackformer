import torch

from stackformer.modules.Attention import Multi_Head_Attention, Self_Attention


def test_self_attention_equivalent_to_single_head_mha_when_weights_are_shared():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 8, requires_grad=True)

    sa = Self_Attention(embed_dim=8, dropout=0.0, qkv_bias=True, mask_type=["causal"])
    mha = Multi_Head_Attention(embed_dim=8, num_heads=1, dropout=0.0, qkv_bias=True, mask_type=["causal"])

    with torch.no_grad():
        mha.q_proj.weight.copy_(sa.q_proj.weight)
        mha.k_proj.weight.copy_(sa.k_proj.weight)
        mha.v_proj.weight.copy_(sa.v_proj.weight)
        if mha.q_proj.bias is not None and sa.q_proj.bias is not None:
            mha.q_proj.bias.copy_(sa.q_proj.bias)
            mha.k_proj.bias.copy_(sa.k_proj.bias)
            mha.v_proj.bias.copy_(sa.v_proj.bias)
        mha.out_proj.weight.copy_(sa.out_proj.weight)
        if mha.out_proj.bias is not None and sa.out_proj.bias is not None:
            mha.out_proj.bias.copy_(sa.out_proj.bias)

    out_sa = sa(x)
    out_mha = mha(x)

    assert out_sa.shape == out_mha.shape == x.shape
    assert torch.isfinite(out_sa).all() and torch.isfinite(out_mha).all()
    assert torch.allclose(out_sa, out_mha, atol=1e-5)
