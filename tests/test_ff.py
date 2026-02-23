import torch

from stackformer.modules.Feed_forward import (
    FF_ReLU,
    FF_LeakyReLU,
    FF_GELU,
    FF_Sigmoid,
    FF_SiLU,
    FF_SwiGLU,
    FF_GeGLU,
)


BATCH = 2
SEQ = 4
EMB = 8
HIDDEN = 16


def _run_ff(ff):
    x = torch.randn(BATCH, SEQ, EMB)
    y = ff(x)
    assert y.shape == x.shape


def test_ff_relu():
    ff = FF_ReLU(EMB, HIDDEN, dropout=0.0)
    _run_ff(ff)


def test_ff_leaky_relu():
    ff = FF_LeakyReLU(EMB, HIDDEN, dropout=0.0)
    _run_ff(ff)


def test_ff_gelu():
    ff = FF_GELU(EMB, HIDDEN, dropout=0.0)
    _run_ff(ff)


def test_ff_sigmoid():
    ff = FF_Sigmoid(EMB, HIDDEN, dropout=0.0)
    _run_ff(ff)


def test_ff_silu():
    ff = FF_SiLU(EMB, HIDDEN, dropout=0.0)
    _run_ff(ff)


def test_ff_swiglu():
    ff = FF_SwiGLU(EMB, HIDDEN, dropout=0.0)
    _run_ff(ff)


def test_ff_geglu():
    ff = FF_GeGLU(EMB, HIDDEN, dropout=0.0)
    _run_ff(ff)