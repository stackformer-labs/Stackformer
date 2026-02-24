"""Positional encoding modules for Stackformer attention blocks.

Transformers are permutation-invariant over tokens unless position information is
injected. This module implements three common strategies:
- AbsolutePositionEmbedding: learned lookup by index
- SinusoidalPositionalEmbedding: deterministic sine/cosine basis
- RoPE: rotary relative-position encoding in complex space

These embeddings are designed to plug into token embeddings or Q/K tensors in
attention layers.
"""

import math
import torch
import torch.nn as nn

class AbsolutePositionEmbedding(nn.Module):
    """Learned absolute positional embedding table.

    Constructor args:
        seq_len (int, required): Maximum supported position index.
        embed_dim (int, required): Positional vector size ``D``.
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): Any tensor with leading shape ``(B, T, ...)``. Only
            ``B`` and ``T`` are used.

    Returns:
        torch.Tensor: Positional vectors of shape ``(B, T, D)``.

    Rule:
        Runtime ``T`` must satisfy ``T <= seq_len``.

    Example:
        >>> pos_emb = AbsolutePositionEmbedding(seq_len=512, embed_dim=128)
        >>> x = torch.randn(4, 32, 128)
        >>> pos = pos_emb(x)
        >>> pos.shape
        torch.Size([4, 32, 128])
    
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
    """Deterministic sinusoidal positional encoding.

    Formula for position ``p`` and channel pair ``i``:
        PE[p, 2i]   = sin(p / 10000^(2i/D))
        PE[p, 2i+1] = cos(p / 10000^(2i/D))

    Constructor args:
        seq_len (int, required): Maximum precomputed sequence length.
        embed_dim (int, required): Encoding size ``D``.
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): Shape ``(B, T, ...)`` where only ``B`` and ``T`` are
            consumed.

    Returns:
        torch.Tensor: Shape ``(B, T, D)``.

    Rule:
        Runtime ``T`` must satisfy ``T <= seq_len`` (buffer is precomputed).

    Example:
        >>> pos_emb = SinusoidalPositionalEmbedding(seq_len=512, embed_dim=128)
        >>> x = torch.randn(4, 32, 128)
        >>> pos = pos_emb(x)
        >>> pos.shape
        torch.Size([4, 32, 128])
    
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

        self.register_buffer("pe", pe.to(device=self.device, dtype=self.dtype))

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        out = self.pe[:seq_len].unsqueeze(0).expand(batch_size, seq_len, -1)
        return out.to(device=self.device, dtype=self.dtype)

class RoPE(nn.Module):
    """Rotary positional embedding for attention query/key tensors.

    RoPE rotates each 2D pair in head dimension by a position-dependent angle,
    encoding relative position directly into dot-product attention.

    Constructor args:
        head_dim (int, required): Per-head size ``D``. Rule: must be even.
        seq_len (int, required): Maximum supported sequence length.
        theta (float, optional, default=10000.0): Frequency base.
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): Shape ``(B, H, T, D)``.

    Returns:
        torch.Tensor: Shape ``(B, H, T, D)`` with rotary transform applied.

    Rules:
        - ``D`` must be even.
        - Runtime ``T`` must satisfy ``T <= seq_len``.

    Example:
        >>> rope = RoPE(head_dim=64, seq_len=512)
        >>> q = torch.randn(2, 8, 32, 64)
        >>> q_rot = rope(q)
        >>> q_rot.shape
        torch.Size([2, 8, 32, 64])
    """
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
