import torch

from stackformer.modules.Feed_forward import FF_GELU, FF_GeGLU, FF_LeakyReLU, FF_ReLU, FF_SiLU, FF_Sigmoid, FF_SwiGLU


def test_feedforward_variants_shape():
    x = torch.randn(2, 4, 8)
    for ff in (
        FF_ReLU(8, 16, dropout=0.0),
        FF_LeakyReLU(8, 16, dropout=0.0),
        FF_GELU(8, 16, dropout=0.0),
        FF_Sigmoid(8, 16, dropout=0.0),
        FF_SiLU(8, 16, dropout=0.0),
        FF_SwiGLU(8, 16, dropout=0.0),
        FF_GeGLU(8, 16, dropout=0.0),
    ):
        assert ff(x).shape == x.shape
