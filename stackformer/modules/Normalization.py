"""Normalization layers for Stackformer blocks.

This module provides two per-token normalization operators used in transformer
architectures:
- LayerNormalization (mean+variance normalization with affine scale/bias)
- RMSNormalization (root-mean-square normalization with scale only)

Given ``x`` with last dimension ``C``:
- LayerNorm: ``y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta``
- RMSNorm:   ``y = gamma * x / (sqrt(mean(x^2)) + eps)``

Notation:
- ``B``: batch size
- ``T``: sequence length
- ``C``: embedding dimension (normalized dimension)
"""

import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    """Layer normalization over the last tensor dimension.

    Constructor args:
        embed_dim (int, required): Normalized feature size ``C``.
        eps (float, optional, default=1e-5): Numerical stability constant.
        device (str or torch.device, optional, default='cpu').
        dtype (torch.dtype, optional, default=torch.float32).

    Learnable parameters:
        weight (gamma): Shape ``(C,)``.
        bias (beta): Shape ``(C,)``.

    Forward args:
        x (torch.Tensor): Shape ``(..., C)``.

    Returns:
        torch.Tensor: Same shape as input ``(..., C)``.

    Example:
        >>> norm = LayerNormalization(embed_dim=256, eps=1e-5)
        >>> x = torch.randn(4, 32, 256)
        >>> y = norm(x)
        >>> y.shape
        torch.Size([4, 32, 256])
    
    """
    def __init__(self, embed_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(embed_dim,device=self.device,dtype=self.dtype))  # gamma
        self.bias = nn.Parameter(torch.zeros(embed_dim,device=self.device,dtype=self.dtype))   # beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True).to(self.device)
        var = x.var(dim=-1, keepdim=True, unbiased=False).to(self.device)
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)
        output = self.weight * normalized_x + self.bias
        return output.to(device=self.device, dtype=self.dtype)

class RMSNormalization(nn.Module):
    """RMSNorm over the last tensor dimension (no mean subtraction).

    Constructor args:
        embed_dim (int, required): Feature size ``C``.
        eps (float, optional, default=1e-5): Numerical stability constant.
        device (str or torch.device, optional, default='cpu').
        dtype (torch.dtype, optional, default=torch.float32).

    Learnable parameters:
        weight (gamma): Shape ``(C,)``.

    Forward args:
        x (torch.Tensor): Shape ``(..., C)``.

    Returns:
        torch.Tensor: Same shape as input ``(..., C)``.

    Practical note:
        RMSNorm is often faster and slightly more stable for large language models.

    Example:
        >>> norm = RMSNormalization(embed_dim=256)
        >>> x = torch.randn(4, 32, 256)
        >>> y = norm(x)
        >>> y.shape
        torch.Size([4, 32, 256])
    
    """
    def __init__(self, embed_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(embed_dim, device=self.device,dtype=self.dtype))  # gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).sqrt().to(device=self.device)
        normalized_x = x / (rms + self.eps).to(device=self.device)
        output = self.weight * normalized_x
        return output.to(device=self.device, dtype=self.dtype)
