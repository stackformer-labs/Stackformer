import torch

from stackformer.modules.Normalization import LayerNormalization, RMSNormalization


BATCH = 2
SEQ = 4
EMB = 8


def test_layer_normalization_shape():
    x = torch.randn(BATCH, SEQ, EMB)

    ln = LayerNormalization(embed_dim=EMB)
    y = ln(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_rms_normalization_shape():
    x = torch.randn(BATCH, SEQ, EMB)

    rms = RMSNormalization(embed_dim=EMB)
    y = rms(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_layer_norm_zero_input():
    x = torch.zeros(BATCH, SEQ, EMB)

    ln = LayerNormalization(embed_dim=EMB)
    y = ln(x)

    # should not produce NaNs
    assert torch.isfinite(y).all()


def test_rms_norm_zero_input():
    x = torch.zeros(BATCH, SEQ, EMB)

    rms = RMSNormalization(embed_dim=EMB)
    y = rms(x)

    assert torch.isfinite(y).all()