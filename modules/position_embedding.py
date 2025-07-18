import torch
import torch.nn as nn
import math

class AbsolutePositionEmbedding(nn.Module):
    def __init__(self, seq_len, emb_dim):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim

        # Initialize absolute position indices and embed them
        self.position = nn.Parameter(torch.arange(0, seq_len).unsqueeze(1).float(), requires_grad=False)
        self.embedding = nn.Embedding(seq_len, emb_dim)

    def forward(self, batch_size=None):
        pos_emb = self.embedding(self.position.squeeze(1).long())  # (T, D)
        if batch_size is not None:
            return pos_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (B, T, D)
        else:
            return pos_emb  # (T, D)

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, seq_len, emb_dim):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim

        position = torch.arange(0, seq_len).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))  # (emb_dim/2)

        pe = torch.zeros(seq_len, emb_dim)  # (seq_len, emb_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it's moved with model
        self.register_buffer("pe", pe)

    def forward(self, batch_size=None):
        # Return with shape (B, T, D) if batch_size is given
        if batch_size is not None:
            return self.pe.unsqueeze(0).expand(batch_size, -1, -1)  # (B, T, D)
        else:
            return self.pe  # (T, D)
