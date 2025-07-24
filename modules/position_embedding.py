import torch
import torch.nn as nn
import math

# --- Absolute Positional Embedding ---
class AbsolutePositionEmbedding(nn.Module):
    def __init__(self, seq_len, emb_dim):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(seq_len, emb_dim)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        positions = torch.arange(0, seq_len)
        abs_pos = self.embedding(positions)  # (seq_len, emb_dim)
        return abs_pos.unsqueeze(0).expand(batch_size, seq_len, -1).to(x.device)

# --- Sinusoidal Positional Embedding ---
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, seq_len, emb_dim):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim

        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))

        pe = torch.zeros(seq_len, emb_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, emb_dim) or (batch_size, seq_len)
        batch_size, seq_len = x.shape[0], x.shape[1]
        return self.pe[:seq_len].unsqueeze(0).expand(batch_size, seq_len, -1).to(x.device)
    
# --- RoPE ---
class RoPE(nn.Module):
    def __init__(self, head_dim, seq_len, theta=10000.0, device='cpu', dtype=torch.float32):
        super().__init__()
        self.dtype  = dtype
        self.device = device
        assert head_dim % 2 == 0, "head_dim must be even"
        theta_numerator = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
        inv_freq = 1.0 / (theta ** (theta_numerator / head_dim))
        m = torch.arange(seq_len, device=device)
        freqs = torch.outer(m, inv_freq)
        self.register_buffer("freq_complex", torch.polar(torch.ones_like(freqs), freqs)) 

    def forward(self, x):
        batch_size, seq_len, num_head, emb_dim = x.shape
        assert emb_dim % 2 == 0, "emb_dim must be even"
        x_reshaped = x.view(batch_size, seq_len, num_head, emb_dim // 2, 2)
        x_complex = torch.view_as_complex(x_reshaped)
        freqs = self.freq_complex[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)
        x_rotated = x_complex * freqs
        x_out = torch.view_as_real(x_rotated).contiguous().view(batch_size, seq_len, num_head, emb_dim)
        return x_out.to(device=self.device, dtype=self.dtype)