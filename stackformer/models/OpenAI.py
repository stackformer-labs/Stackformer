import torch
import torch.nn as nn

from stackformer.modules.Attention import Multi_Head_Attention
from stackformer.modules.position_embedding import AbsolutePositionEmbedding
from stackformer.modules.Normalization import LayerNormalization
from stackformer.modules.Feed_forward import FF_GELU
from stackformer.generate import text_generate


'''
GPT-1
Attention: MHA
Mask: Causal
Position: Absolute
FF: GeLU
Norm: Post LayerNorm
'''

# ---------------- GPT-1 Block ----------------

class GPT_1_Block(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout,
        hidden_dim,
        qkv_bias=True,
        eps=1e-5,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()

        self.attention = Multi_Head_Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            qkv_bias=qkv_bias,
            device=device,
            dtype=dtype,
        )

        self.norm1 = LayerNormalization(embed_dim, eps=eps, device=device, dtype=dtype)
        self.ff = FF_GELU(embed_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm2 = LayerNormalization(embed_dim, eps=eps, device=device, dtype=dtype)

    def forward(self, x):
        residual = x
        x = self.attention(x)
        x = self.norm1(x)
        x = x + residual

        residual = x
        x = self.ff(x)
        x = self.norm2(x)
        x = x + residual

        return x


# ---------------- GPT-1 Decoder ----------------

class GPT_1_Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        dropout,
        hidden_dim,
        qkv_bias=True,
        eps=1e-5,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                GPT_1_Block(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    hidden_dim=hidden_dim,
                    qkv_bias=qkv_bias,
                    eps=eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GPT_1(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_layers,
        embed_dim,
        num_heads,
        seq_len,
        dropout,
        hidden_dim,
        qkv_bias=True,
        eps=1e-5,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.seq_len = seq_len

        self.embedding = nn.Embedding(vocab_size, embed_dim, device=device, dtype=dtype)
        self.position_embedding = AbsolutePositionEmbedding(seq_len, embed_dim, device=device, dtype=dtype)

        self.decoder = GPT_1_Decoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            hidden_dim=hidden_dim,
            qkv_bias=qkv_bias,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        self.final_norm = LayerNormalization(embed_dim, eps=eps, device=device, dtype=dtype)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        emb = self.embedding(x)
        pos = self.position_embedding(x)
        x = emb + pos
        x = self.decoder(x)
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return text_generate(self, *args, **kwargs)


'''
GPT-2
Attention: MHA
Norm: Pre LayerNorm
'''

# ---------------- GPT-2 Block ----------------

class GPT_2_Block(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout,
        hidden_dim,
        qkv_bias=True,
        eps=1e-5,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()

        self.attention = Multi_Head_Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            qkv_bias=qkv_bias,
            device=device,
            dtype=dtype,
        )

        self.norm1 = LayerNormalization(embed_dim, eps=eps, device=device, dtype=dtype)
        self.ff = FF_GELU(embed_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm2 = LayerNormalization(embed_dim, eps=eps, device=device, dtype=dtype)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + residual

        return x


# ---------------- GPT-2 Decoder ----------------

class GPT_2_Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        dropout,
        hidden_dim,
        qkv_bias=True,
        eps=1e-5,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                GPT_2_Block(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    hidden_dim=hidden_dim,
                    qkv_bias=qkv_bias,
                    eps=eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GPT_2(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_layers,
        embed_dim,
        num_heads,
        seq_len,
        dropout,
        hidden_dim,
        qkv_bias=True,
        eps=1e-5,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, device=device, dtype=dtype)
        self.position_embedding = AbsolutePositionEmbedding(seq_len, embed_dim, device=device, dtype=dtype)

        self.decoder = GPT_2_Decoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            hidden_dim=hidden_dim,
            qkv_bias=qkv_bias,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        self.final_norm = LayerNormalization(embed_dim, eps=eps, device=device, dtype=dtype)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        emb = self.embedding(x)
        pos = self.position_embedding(x)
        x = emb + pos
        x = self.decoder(x)
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return text_generate(self, *args, **kwargs)
