import pytest
import torch

from stackformer.modules.Attention import kv_cache_group_query, kv_cache_multihead

BATCH = 2
SEQ = 6
EMB = 8
HEADS = 2


@pytest.mark.parametrize("cache_cls,kwargs,kv_heads", [
    (kv_cache_multihead, dict(embed_dim=EMB, num_heads=HEADS, batch_size=BATCH, kv_seq_len=SEQ, dropout=0.0), HEADS),
    (
        kv_cache_group_query,
        dict(embed_dim=EMB, num_query_heads=HEADS, num_kv_heads=1, batch_size=BATCH, kv_seq_len=SEQ, dropout=0.0),
        1,
    ),
])
def test_kv_cache_growth_per_step_and_no_overwrite(cache_cls, kwargs, kv_heads):
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, EMB)
    cache = cache_cls(**kwargs)

    first_written = []
    for step in range(SEQ):
        token = x[:, step : step + 1]
        out = cache(token, start_pos=step, rope=False)
        assert out.shape == (BATCH, 1, EMB)
        assert torch.isfinite(out).all()

        current = cache.cache_keys[:, : step + 1].detach().clone()
        assert current.shape == (BATCH, step + 1, kv_heads, EMB // HEADS)

        if step == 0:
            first_written = current.clone()
        else:
            assert torch.allclose(cache.cache_keys[:, :1], first_written)
