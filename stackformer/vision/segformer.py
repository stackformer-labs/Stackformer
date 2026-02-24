import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""SegFormer-B0 implementation for semantic segmentation.

This module contains a compact SegFormer-style encoder-decoder model with a
multi-scale transformer encoder and a lightweight MLP decoder.

Paper reference: https://arxiv.org/abs/2105.15203
"""

# Patch embedding layer
class patch(nn.Module):
    def __init__(self, img_size=224, in_channels=3, out_channels=768, kernel=7, stride=3, padding=3):
        super().__init__()
        self.img_size = img_size
        self.kernel = kernel
        self.out_channels = out_channels
        self.num_patches = (img_size // kernel) ** 2
        
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels, 
                kernel_size=kernel, stride=stride, padding=padding)
        
    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        return x

# Efficient Multi Head Attention (MHA)
class Multi_Head_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, reduction=1, device='cpu', dtype=torch.float32):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.device = device
        self.dtype = dtype
        self.reduction = reduction

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True, device=device, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True, device=device, dtype=dtype)
        
        # Downsampling for K and V (if reduction > 1)
        if reduction > 1:
            self.kv_down = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=reduction, padding=1, groups=embed_dim)
        else:
            self.kv_down = nn.Identity()
        
        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        x = x.to(device=self.device, dtype=self.dtype)

        q = self.q_proj(x)  # (B, N, C)
        
        if self.reduction > 1:
            x_down = x.transpose(1, 2)  # (B, C, N)
            x_down = self.kv_down(x_down).transpose(1, 2)  # (B, C, N_reduced) => (B, N_reduced, C)
            
            k = self.k_proj(x_down)  # (B, N_reduced, C)
            v = self.v_proj(x_down)  # (B, N_reduced, C)
        else:
            k = self.k_proj(x)  # (B, N, C)
            v = self.v_proj(x)  # (B, N, C)
        
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, q.size(2), C)

        return self.out_proj(out)

# Feed forward layer
class FF_GELU(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0, device='cpu', dtype=torch.float32):
        super().__init__()
        
        self.gelu = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=True,device=device, dtype=dtype),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, bias=True,device=device, dtype=dtype),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.gelu(x)

# Transformer encoding block
class transformer_block(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout, reduction):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.att = Multi_Head_Attention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, reduction=reduction)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FF_GELU(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout)
    
    def forward(self, x):
        x = x + self.att(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

# Transformer encoding layer
class transformer_encoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, hidden_dim, dropout, reduction):
        super().__init__()
        self.layers = nn.ModuleList([
            transformer_block(embed_dim, num_heads, hidden_dim, dropout, reduction)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # SegFormer B0 configuration
        self.s1 = patch(img_size=224, in_channels=3, out_channels=32,
                        kernel=7, stride=4, padding=3)
        self.te1 = transformer_encoder(
            num_layers=2, embed_dim=32, num_heads=1,
            hidden_dim=32 * 8, dropout=0.0, reduction=8
        )

        self.s2 = patch(img_size=56, in_channels=32, out_channels=64,
                        kernel=3, stride=2, padding=1)
        self.te2 = transformer_encoder(
            num_layers=2, embed_dim=64, num_heads=2,
            hidden_dim=64 * 4, dropout=0.0, reduction=4
        )

        self.s3 = patch(img_size=28, in_channels=64, out_channels=160,
                        kernel=3, stride=2, padding=1)
        self.te3 = transformer_encoder(
            num_layers=2, embed_dim=160, num_heads=5,
            hidden_dim=160 * 4, dropout=0.0, reduction=2
        )

        self.s4 = patch(img_size=14, in_channels=160, out_channels=256,
                        kernel=3, stride=2, padding=1)
        self.te4 = transformer_encoder(
            num_layers=2, embed_dim=256, num_heads=8,
            hidden_dim=256 * 4, dropout=0.0, reduction=1
        )

    def forward(self, x):

        # ---------------- Stage 1 ----------------
        x1 = self.s1(x)          # (B, N1, 32)
        f1 = self.te1(x1)

        B, N1, C1 = f1.shape
        H1 = W1 = int(N1 ** 0.5)
        f1_spatial = f1.transpose(1, 2).reshape(B, C1, H1, W1)

        # ---------------- Stage 2 ----------------
        x2 = self.s2(f1_spatial)  # (B, N2, 64)
        f2 = self.te2(x2)

        B, N2, C2 = f2.shape
        H2 = W2 = int(N2 ** 0.5)
        f2_spatial = f2.transpose(1, 2).reshape(B, C2, H2, W2)

        # ---------------- Stage 3 ----------------
        x3 = self.s3(f2_spatial)  # (B, N3, 160)
        f3 = self.te3(x3)

        B, N3, C3 = f3.shape
        H3 = W3 = int(N3 ** 0.5)
        f3_spatial = f3.transpose(1, 2).reshape(B, C3, H3, W3)

        # ---------------- Stage 4 ----------------
        x4 = self.s4(f3_spatial)  # (B, N4, 256)
        f4 = self.te4(x4)

        B, N4, C4 = f4.shape
        H4 = W4 = int(N4 ** 0.5)
        f4_spatial = f4.transpose(1, 2).reshape(B, C4, H4, W4)

        return f1_spatial, f2_spatial, f3_spatial, f4_spatial

class MLP(nn.Module):
    def __init__(self, out_dim=150):  # 150 classes for ADE20K
        super().__init__()
        
        # Project all features to same channel dimension
        self.f1_proj = nn.Linear(32, 256)
        self.f2_proj = nn.Linear(64, 256)
        self.f3_proj = nn.Linear(160, 256)
        self.f4_proj = nn.Linear(256, 256)
        
        # Final classification head
        self.classifier = nn.Linear(256 * 4, out_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, features):
        f1, f2, f3, f4 = features
        B = f1.shape[0]
        
        # Get target size (highest resolution)
        target_size = f1.shape[2:]  # (H1, W1)
        
        # Upsample all features to same spatial size
        f2_up = F.interpolate(f2, size=target_size, mode='bilinear', align_corners=False)
        f3_up = F.interpolate(f3, size=target_size, mode='bilinear', align_corners=False)
        f4_up = F.interpolate(f4, size=target_size, mode='bilinear', align_corners=False)
        
        # Project to same channel dimension
        # Convert to (B, H*W, C) for linear layers
        f1_flat = f1.flatten(2).transpose(1, 2)  # (B, H*W, 32)
        f2_flat = f2_up.flatten(2).transpose(1, 2)  # (B, H*W, 64)
        f3_flat = f3_up.flatten(2).transpose(1, 2)  # (B, H*W, 160)
        f4_flat = f4_up.flatten(2).transpose(1, 2)  # (B, H*W, 256)
        
        f1_proj = self.f1_proj(f1_flat)  # (B, H*W, 256)
        f2_proj = self.f2_proj(f2_flat)  # (B, H*W, 256)
        f3_proj = self.f3_proj(f3_flat)  # (B, H*W, 256)
        f4_proj = self.f4_proj(f4_flat)  # (B, H*W, 256)
        
        # Concatenate features
        fused = torch.cat([f1_proj, f2_proj, f3_proj, f4_proj], dim=-1)  # (B, H*W, 1024)
        
        # Apply dropout and classification
        fused = self.dropout(fused)
        output = self.classifier(fused)  # (B, H*W, out_dim)
        
        # Reshape back to spatial format
        H, W = target_size
        output = output.transpose(1, 2).reshape(B, -1, H, W)  # (B, out_dim, H, W)
        
        return output

class SegFormerB0(nn.Module):
    """SegFormer-B0 style model for semantic segmentation.

    Simple explanation:
        SegFormer takes an input image and builds feature maps at multiple
        scales (from high resolution to low resolution) using hierarchical
        transformer stages. These features are fused by an MLP decoder to
        produce per-pixel class predictions.

    Architecture details (current implementation):
        - Task: semantic segmentation
        - Encoder type: hierarchical transformer with 4 stages
        - Stage channels: [32, 64, 160, 256]
        - Attention: efficient multi-head self-attention with sequence
          reduction factors [8, 4, 2, 1] across stages
        - Masking: no causal mask (full bidirectional self-attention)
        - Positional strategy: implicit positional cues from overlapping patch
          embeddings (no explicit absolute position embedding tensor)
        - Feed-forward block: GELU MLP per transformer block
        - Normalization: Pre-Norm LayerNorm in each transformer block
        - Decoder: MLP feature projection + multi-scale fusion + bilinear
          upsampling

    Historical context:
        - SegFormer was introduced by NVIDIA in 2021 as a simple and effective
          transformer architecture for semantic segmentation.
        - It avoids heavy decoder designs and explicit positional encodings
          while keeping strong accuracy-speed trade-offs.

    Paper reference:
        - SegFormer paper: https://arxiv.org/abs/2105.15203

    Example:
        >>> import torch
        >>> from stackformer.vision import SegFormerB0
        >>> model = SegFormerB0(num_classes=19)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> y = model(x)
        >>> y.shape
        torch.Size([2, 19, 224, 224])

    Args:
        num_classes: Number of segmentation classes in the output map.
    """
    def __init__(self, num_classes=150):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = MLP(out_dim=num_classes)
        
        # Final projection layer: upsample to original input size
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        # (56x56 → 224x224, since 224/56 = 4)
        
    def forward(self, x):
        H, W = x.shape[2:]  # original size
        features = self.encoder(x)
        output = self.decoder(features)  # (B, num_classes, 56, 56)
        
        # Upsample to match input size
        output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        # print("Final output:", output.shape)
        return output