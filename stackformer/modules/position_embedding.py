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
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        position = torch.arange(0, seq_len).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))  # (D/2)

        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims

        self.register_buffer("pe", pe)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        out = self.pe[:seq_len].unsqueeze(0).expand(batch_size, seq_len, -1)
        return out.to(device=x.device, dtype=x.dtype)

class RoPE(nn.Module):
    """
    Rotary Positional Embedding (RoPE) for attention keys and queries.

    Formula:
        Let x be split into real and imaginary parts as complex numbers:
        - Convert x into complex: x_complex = real + i * imag
        - Apply rotation: x_rotated = x_complex * freq_complex
        - Convert back to real values: out = real(x_rotated), imag(x_rotated)

    Frequencies:
        For position p and dimension d:
            freq[p, d] = 1 / (theta ** (2d / D))

    Args:
        head_dim (int): Per-head embedding dimension (must be even)
        seq_len (int): Maximum sequence length
        theta (float): Base rotation frequency (default: 10000.0)
        device (str): Device to place precomputed values on
        dtype (torch.dtype): Data type for the output

    Input:
        x: Tensor of shape (B, T, H, D), where
            - B: batch size
            - T: sequence length
            - H: number of heads
            - D: head dimension (must be even)

    Output:
        Tensor: Same shape (B, T, H, D)

    Example:
        >>> rope = RoPE(head_dim=64, seq_len=512)
        >>> x = torch.randn(4, 32, 8, 64)  # B=4, T=32, H=8, D=64
        >>> out = rope(x)  # (4, 32, 8, 64)
    """
    def __init__(self, head_dim, seq_len, theta=10000.0, device='cpu', dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.device = device
        assert head_dim % 2 == 0, "head_dim must be even"

        theta_numerator = torch.arange(0, head_dim, 2, device=device)
        inv_freq = 1.0 / (theta ** (theta_numerator / head_dim))  # (D/2)

        m = torch.arange(seq_len, device=device)  # (T)
        freqs = torch.outer(m, inv_freq)  # (T, D/2)
        self.register_buffer("freq_complex", torch.polar(torch.ones_like(freqs), freqs))  # (T, D/2)

    def forward(self, x):
        batch_size, seq_len, num_head, embed_dim = x.shape
        assert embed_dim % 2 == 0, "embed_dim must be even"

        x_reshaped = x.view(batch_size, seq_len, num_head, embed_dim // 2, 2)
        x_complex = torch.view_as_complex(x_reshaped)  # (B, T, H, D/2)

        freqs = self.freq_complex[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, T, 1, D/2)
        x_rotated = x_complex * freqs  # Element-wise complex multiplication

        x_out = torch.view_as_real(x_rotated).contiguous().view(batch_size, seq_len, num_head, embed_dim)
        return x_out.to(device=x.device, dtype=x.dtype)