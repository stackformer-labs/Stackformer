import torch

# Adjust imports to your actual paths if needed
from stackformer.vision.vit import ViT
from stackformer.vision.segformer import (
    patch,
    Multi_Head_Attention,
    transformer_block,
    Encoder,
    SegFormerB0,
)


def test_vit_forward():
    B = 2
    img = torch.randn(B, 3, 32, 32)

    model = ViT(
        img_size=32,
        patch_size=8,
        num_layers=1,
        Emb_dim=32,
        num_heads=4,
        hidden_dim=64,
        num_classes=10,
        dropout=0.0,
    )

    out = model(img)
    assert out.shape == (B, 10)


def test_patch_embedding():
    B = 2
    img = torch.randn(B, 3, 32, 32)

    p = patch(img_size=32, in_channels=3, out_channels=16, kernel=4, stride=4, padding=0)
    out = p(img)

    # tokens = (32 / 4)^2 = 64
    assert out.shape == (B, 64, 16)


def test_segformer_attention():
    B, N, C = 2, 16, 32
    x = torch.randn(B, N, C)

    att = Multi_Head_Attention(embed_dim=C, num_heads=4, dropout=0.0)
    out = att(x)

    assert out.shape == x.shape


def test_transformer_block():
    B, N, C = 2, 16, 32
    x = torch.randn(B, N, C)

    block = transformer_block(
        embed_dim=C,
        num_heads=4,
        hidden_dim=64,
        dropout=0.0,
        reduction=1,
    )

    out = block(x)
    assert out.shape == x.shape


def test_segformer_encoder():
    B = 2
    img = torch.randn(B, 3, 64, 64)

    enc = Encoder()
    f1, f2, f3, f4 = enc(img)

    assert f1.ndim == 4
    assert f2.ndim == 4
    assert f3.ndim == 4
    assert f4.ndim == 4


def test_segformer_full_forward():
    B = 1
    img = torch.randn(B, 3, 64, 64)

    model = SegFormerB0(num_classes=5)
    out = model(img)

    assert out.shape == (B, 5, 64, 64)