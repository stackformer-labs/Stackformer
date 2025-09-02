"""
Positional Embeddings for StackFormer Library

This module implements various positional encoding techniques that inject positional 
information into input embeddings for transformer architectures. Included are:

1. AbsolutePositionalEmbedding
    - Learns a unique embedding vector for each position up to a fixed maximum.
    - Simple and effective for fixed-length input sequences.

2. SinusoidalPositionalEmbedding
    - Uses deterministic sine and cosine functions of different frequencies.
    - Generalizes to longer sequences without additional parameters.

3. RotaryPositionalEmbedding (RoPE)
    - Encodes relative positional information by rotating query/key vectors in complex space.
    - Supports extrapolation and improves attention patterns in autoregressive models.

Each method is implemented in PyTorch and designed to be easily pluggable into 
transformer-based models.
"""

import math
import torch
import torch.nn as nn

class AbsolutePositionEmbedding(nn.Module):
    """
    Learnable absolute positional embedding using a standard embedding layer.

    Formula:
        For each position p in sequence length:
            - Learn a unique embedding vector: embedding(p) ∈ R^D
        For input x of shape (B, T), where B = batch size, T = sequence length:
            - Output shape: (B, T, D)

    Args:
        seq_len (int): Maximum sequence length
        embed_dim (int): Embedding dimension (D)

    Forward Args:
        x (Tensor): Shape (B, T, ...). Only x.shape[0] and x.shape[1] are used.

    Returns:
        Tensor: Positional embeddings of shape (B, T, D)

    Example:
        >>> pos_emb = AbsolutePositionEmbedding(seq_len=512, embed_dim=128)
        >>> x = torch.randn(4, 32, 128)  # (B=4, T=32, D=128)
        >>> pos = pos_emb(x)  # (4, 32, 128)
    """
    def __init__(self, seq_len, embed_dim, device='cpu', dtype=torch.float32):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.device = device
        self.embedding = nn.Embedding(seq_len, embed_dim, device=device, dtype=dtype)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        positions = torch.arange(seq_len, device=self.device, dtype=torch.long)
        abs_pos = self.embedding(positions)  # (seq_len, embed_dim)
        out = abs_pos.unsqueeze(0).expand(batch_size, seq_len, -1)  # (batch, seq_len, embed_dim)
        return out

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Fixed sinusoidal positional embedding used in the original Transformer.

    Formula:
        For each position p and dimension i:
            - PE[p, 2i] = sin(p / (10000^(2i / D)))
            - PE[p, 2i+1] = cos(p / (10000^(2i / D)))

    Input:
        x of shape (B, T, D) or (B, T)

    Output:
        Positional encoding tensor of shape (B, T, D)

    Args:
        seq_len (int): Maximum sequence length
        embed_dim (int): Embedding dimension (must be even)

    Returns:
        Tensor: Sinusoidal encoding (B, T, D)

    Example:
        >>> pe = SinusoidalPositionalEmbedding(seq_len=512, embed_dim=128)
        >>> x = torch.randn(4, 32, 128)
        >>> pos = pe(x)  # (4, 32, 128)
    """
    def __init__(self, seq_len, embed_dim, device='cpu', dtype=torch.float32):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.device=device
        self.dtype = dtype

        position = torch.arange(0, seq_len).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))  # (D/2)

        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims

        self.register_buffer("pe", pe, device=self.device, dtype=self.dtype)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        out = self.pe[:seq_len].unsqueeze(0).expand(batch_size, seq_len, -1)
        return out.to(device=self.device, dtype=self.dtype)

class RoPE(nn.Module):
    def __init__(self, head_dim: int, seq_len: int, theta: float = 10000.0, device="cpu", dtype=torch.float32):
        super().__init__()
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype

        # Precompute and register buffer
        freq_complex = self._precompute_theta_position_frequency(head_dim, seq_len, theta)
        self.register_buffer("freq_complex", freq_complex, persistent=True)

    def _precompute_theta_position_frequency(self, head_dim: int, seq_len: int, theta: float):
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        dim_half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim_half, device=self.device) / dim_half))
        pos = torch.arange(seq_len, device=self.device)
        freqs = torch.outer(pos, inv_freq)  # (seq_len, dim_half)
        freq_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freq_complex  # (seq_len, dim_half)

    def forward(self, x: torch.Tensor):
        B, H, T, D = x.shape
        assert D % 2 == 0, "head_dim must be even for RoPE"

        # reshape to complex
        x = x.view(B, H, T, D // 2, 2)
        x_complex = torch.view_as_complex(x)  # (B, H, T, D//2)

        # slice correct freqs
        freqs = self.freq_complex[:T].unsqueeze(0).unsqueeze(0)  # (1,1,T,D//2)

        # apply rotation
        x_rot = x_complex * freqs  # (B, H, T, D//2)

        # back to real
        x_out = torch.view_as_real(x_rot).view(B, H, T, D)
        return x_out.to(dtype=self.dtype, device=x.device)