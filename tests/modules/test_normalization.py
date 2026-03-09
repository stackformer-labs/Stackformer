import torch

from stackformer.modules.Normalization import LayerNormalization, RMSNormalization


def test_normalization_outputs_are_finite():
    x = torch.randn(2, 4, 8)
    assert torch.isfinite(LayerNormalization(embed_dim=8)(x)).all()
    assert torch.isfinite(RMSNormalization(embed_dim=8)(x)).all()


def test_normalization_zero_input_stable():
    x = torch.zeros(2, 4, 8)
    assert torch.isfinite(LayerNormalization(embed_dim=8)(x)).all()
    assert torch.isfinite(RMSNormalization(embed_dim=8)(x)).all()
