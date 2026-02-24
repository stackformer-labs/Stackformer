import torch
import torch.nn as nn

from stackformer.modules.Attention import Multi_Head_Attention
from stackformer.modules.Feed_forward import FF_ReLU

"""Vision Transformer (ViT) building blocks.

This file implements a classification-focused Vision Transformer based on the
paper architecture introduced in 2020.

Paper (add your link): [TODO: add paper link]
"""

# Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, emb_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.proj(x)                 # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2) # [B, N, D]
        return x

# Transformer Block
class Block(nn.Module):
    def __init__(self, Emb_dim, num_heads, dropout, hidden_dim, eps=1e-5, device=None, dtype=torch.float32):
        super().__init__()
        # Don't pass device to submodules - let them be moved with .to() later
        self.attention = Multi_Head_Attention(Emb_dim, num_heads, dropout, qkv_bias=True,device=device, dtype=dtype)
        self.norm1 = nn.LayerNorm(Emb_dim, eps=eps, dtype=dtype)
        self.ff_relu = FF_ReLU(Emb_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(Emb_dim, eps=eps, dtype=dtype)

    def forward(self, x):
        # Pre-norm attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask=False)
        x = x + residual

        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = self.ff_relu(x)
        x = x + residual

        return x

# Encoder (stack of blocks)
class Encoder(nn.Module):
    def __init__(self, num_layers, Emb_dim, num_heads, dropout, hidden_dim, eps=1e-5, device=None, dtype=torch.float32):
        super().__init__()
        self.layers = nn.ModuleList([
            Block(Emb_dim, num_heads, dropout, hidden_dim, eps, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Vision Transformer (ViT)
class ViT(nn.Module):
    """Vision Transformer for image classification.

    Simple explanation:
        ViT splits an image into fixed-size patches, projects each patch into an
        embedding vector, and processes the patch sequence with transformer
        encoder blocks. A learnable ``[CLS]`` token is prepended, and its final
        representation is used for classification.

    Architecture details (current implementation):
        - Task: image classification
        - Input tokenization: non-overlapping convolutional patch projection
        - Attention: multi-head self-attention (full global attention)
        - Masking: no causal mask (bidirectional attention over all patches)
        - Positional encoding: learnable absolute positional embeddings
        - Feed-forward block: ReLU-based FFN (``FF_ReLU``)
        - Normalization: Pre-Norm LayerNorm inside each transformer block
        - Head: LayerNorm + Linear classifier on ``[CLS]`` token

    Historical context:
        - Model family introduced by the paper "An Image is Worth 16x16 Words"
          (Google Research, 2020; published at ICLR 2021).
        - ViT showed that transformer-only backbones can match or exceed strong
          CNN baselines when trained with sufficient data.

    Paper reference:
        - ViT paper: [TODO: add paper link]

    Example:
        >>> import torch
        >>> from stackformer.vision import ViT
        >>> model = ViT(img_size=224, patch_size=16, num_classes=1000)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([2, 1000])

    Args:
        img_size: Input image height/width (square expected).
        patch_size: Patch size used for tokenization.
        num_layers: Number of transformer encoder layers.
        Emb_dim: Embedding dimension for patch tokens.
        num_classes: Number of output classes.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        hidden_dim: Hidden dimension of the feed-forward layer.
        eps: Epsilon used by layer normalization.
    """
    def __init__(self, img_size=224, patch_size=16, num_layers=12,
                 Emb_dim=768, num_classes=1000, num_heads=12,
                 dropout=0.1, hidden_dim=3072, eps=1e-5):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, Emb_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, Emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embedding.num_patches, Emb_dim))
        self.dropout = nn.Dropout(dropout)

        # Encoder - don't pass device here, let .to() handle it
        self.encoder = Encoder(num_layers, Emb_dim, num_heads, dropout, hidden_dim, eps)

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(Emb_dim, eps=eps),
            nn.Linear(Emb_dim, num_classes)
        )

        # Init weights
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embedding(x)   # [B, N, D]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        pos_embed = self.pos_embed

        x = torch.cat((cls_tokens, x), dim=1)   # [B, 1+N, D]
        x = x + pos_embed
        x = self.dropout(x)

        x = self.encoder(x)
        return self.mlp_head(x[:, 0])   # [B, num_classes]
