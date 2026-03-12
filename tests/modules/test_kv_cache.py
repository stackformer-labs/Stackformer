import pytest
import torch

from stackformer.modules.Attention import kv_cache_group_query, kv_cache_multihead


@pytest.mark.parametrize("batch", [1, 2])
def test_kv_cache_multihead_updates_cache_and_stays_stable(batch):
    torch.manual_seed(0)
    layer = kv_cache_multihead(embed_dim=8, num_heads=2, batch_size=2, kv_seq_len=6, dropout=0.0)

    x1 = torch.randn(batch, 2, 8)
    y1 = layer(x1, start_pos=0, rope=False)
    x2 = torch.randn(batch, 1, 8)
    y2 = layer(x2, start_pos=2, rope=False)

    assert y1.shape == x1.shape
    assert y2.shape == x2.shape
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()
    assert torch.abs(y2).mean() < 100
    assert layer.cache_keys[:batch, :3].shape == (batch, 3, 2, 4)


def test_kv_cache_group_query_updates_cache_and_stays_stable():
    torch.manual_seed(0)
    layer = kv_cache_group_query(embed_dim=8, num_query_heads=2, num_kv_heads=1, batch_size=2, kv_seq_len=6, dropout=0.0)

    x1 = torch.randn(2, 2, 8)
    y1 = layer(x1, start_pos=0, rope=False)
    x2 = torch.randn(2, 1, 8)
    y2 = layer(x2, start_pos=2, rope=False)

    assert y1.shape == x1.shape
    assert y2.shape == x2.shape
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()
    assert torch.abs(y2).mean() < 100
    assert layer.cache_keys[:, :3].shape == (2, 3, 1, 4)
