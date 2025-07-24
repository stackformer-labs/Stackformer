import torch
import torch.nn as nn
import math

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
