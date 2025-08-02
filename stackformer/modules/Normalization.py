import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    """
    Applies standard Layer Normalization over the last dimension of the input tensor.

    Formula:
        Input: x of shape (batch_size, seq_length, embedding_dim)
        - Compute mean along the last dimension: mean = average(x)
        - Compute variance along the last dimension: var = average((x - mean)²)
        - Normalize: normalized_x = (x - mean) / sqrt(var + epsilon)
        - Scale and shift: output = gamma * normalized_x + beta

    Args:
        embed_dim (int): Size of the last dimension (embedding dimension)
        eps (float): Small constant added to avoid division by zero

    Forward Args:
        x (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim)

    Returns:
        Tensor: Output tensor of the same shape as input

    Example:
        >>> layer_norm = LayerNormalization(embed_dim=64)
        >>> x = torch.randn(4, 10, 64)
        >>> output = layer_norm(x)
    """
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))  # gamma
        self.bias = nn.Parameter(torch.zeros(embed_dim))   # beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)
        output = self.weight * normalized_x + self.bias
        return output.to(device=x.device, dtype=x.dtype)

class RMSNormilization(nn.Module):
    """
    Applies RMS Layer Normalization over the last dimension of the input tensor.

    Formula:
        Input: x of shape (batch_size, seq_length, embedding_dim)
        - Compute root mean square: rms = sqrt(average(x²))
        - Normalize: normalized_x = x / (rms + epsilon)
        - Scale: output = gamma * normalized_x

    Args:
        embed_dim (int): Size of the last dimension (embedding dimension)
        eps (float): Small constant added to avoid division by zero

    Forward Args:
        x (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim)

    Returns:
        Tensor: Output tensor of the same shape as input

    Example:
        >>> rms_norm = RMSNormilization(embed_dim=64)
        >>> x = torch.randn(4, 10, 64)
        >>> output = rms_norm(x)
    """
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))  # gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        normalized_x = x / (rms + self.eps)
        output = self.weight * normalized_x
        return output.to(device=x.device, dtype=x.dtype)