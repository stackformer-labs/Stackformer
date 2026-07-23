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
    def __init__(self, seq_len, embed_dim, device=None, dtype=None):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embedding = nn.Embedding(seq_len, embed_dim, **factory_kwargs)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        # Derive device from the embedding weight so this works correctly after
        # .to(device) without relying on the stale self.device string.
        if seq_len > self.seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.seq_len}."
            )
        positions = torch.arange(seq_len, device=self.embedding.weight.device, dtype=torch.long)
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
    def __init__(self, seq_len, embed_dim, device=None, dtype=None):
        super().__init__()

        if embed_dim % 2 != 0:
            raise ValueError(
                f"embed_dim must be even, got {embed_dim}"
            )

        factory_kwargs = {"device": device, "dtype": dtype}

        position = torch.arange(seq_len, **factory_kwargs).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, **factory_kwargs)* -(math.log(10000.0) / embed_dim))

        pe = torch.zeros(seq_len, embed_dim, **factory_kwargs)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        B, T = x.shape[:2]
        if T > self.pe.size(0):
            raise ValueError(
                f"Sequence length {T} exceeds maximum {self.pe.size(0)}."
            )
        return self.pe[:T].unsqueeze(0).expand(B, T, -1)

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
    def __init__(self, head_dim: int, seq_len: int, device=None, dtype=None, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.theta = theta
        self.factory_kwargs = {"device": device, "dtype": dtype}

        # Precompute and register buffer
        freq_complex = self._precompute_theta_position_frequency(head_dim, seq_len, theta)
        self.register_buffer("freq_complex", freq_complex, persistent=True)

    def _precompute_theta_position_frequency(self, head_dim: int, seq_len: int, theta: float):
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        dim_half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim_half, **self.factory_kwargs) / dim_half))
        pos = torch.arange(seq_len, **self.factory_kwargs)
        freqs = torch.outer(pos, inv_freq)  # (seq_len, dim_half)
        freq_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freq_complex  # (seq_len, dim_half)

    def forward(self, x: torch.Tensor):
        B, H, T, D = x.shape
        if D % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        # reshape to complex
        x_r = x.reshape(B, H, T, D // 2, 2)
        x_complex = torch.view_as_complex(x_r.float())  # (B, H, T, D//2); promote for complex ops

        # slice correct freqs — freq_complex is a registered buffer, already on
        # the correct device after .to(device). Cast to match x_complex.
        if T > self.freq_complex.size(0):
            raise ValueError(
                f"Sequence length {T} exceeds maximum {self.freq_complex.size(0)}."
            )
        freqs = self.freq_complex[:T].unsqueeze(0).unsqueeze(0)  # (1,1,T,D//2)
        freqs = freqs.to(x_complex)

        # apply rotation
        x_rot = x_complex * freqs  # (B, H, T, D//2)

        # back to real, cast back to original dtype
        x_out = torch.view_as_real(x_rot).reshape(B, H, T, D)
        return x_out