import math

import torch

from stackformer.modules.Masking import make_mask

BATCH = 2
SEQ = 6
EMB = 8
WINDOW = 2


def _manual_masked_attention(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    q = x
    k = x
    v = x
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(x.size(-1))
    scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


def test_sliding_window_mask_blocks_out_of_window_tokens():
    torch.manual_seed(0)
    x1 = torch.randn(BATCH, SEQ, EMB)
    x2 = x1.clone()

    # For last token with window=2, index 0 is outside allowed history.
    x2[:, 0] += 10.0

    window_mask = make_mask(["sliding_window"], seq_len=SEQ, window_size=WINDOW)
    out1 = _manual_masked_attention(x1, window_mask)
    out2 = _manual_masked_attention(x2, window_mask)

    assert out1.shape == out2.shape == (BATCH, SEQ, EMB)
    assert torch.isfinite(out1).all() and torch.isfinite(out2).all()

    # Last token output should be unchanged because index 0 is outside window.
    assert torch.allclose(out1[:, -1], out2[:, -1], atol=1e-6)
