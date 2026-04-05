"""OpenAI GPT-family decoder-only model implementations.

This module contains GPT-1 and GPT-2 style language models with concise,
research-aware documentation and practical usage examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        self.norm1 = nn.LayerNorm(embed_dim, eps=eps, device=device, dtype=dtype)
        self.ff = FF_GELU(embed_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(embed_dim, eps=eps, device=device, dtype=dtype)
        
    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ff(x))
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
    """GPT-1 style decoder-only causal language model.

    Simple explanation:
        GPT-1 is an early large-scale autoregressive transformer that predicts
        the next token from previous tokens. This implementation follows the
        same high-level decoder-only design.

    Architecture details:
        - Attention: Multi-Head self-attention.
        - Masking: Causal mask.
        - Position encoding: Learned absolute positional embeddings.
        - Feed-forward: GELU MLP.
        - Normalization: Post-LayerNorm style inside blocks, with final norm.

    Research context:
        - Historical role: established transfer learning with generative
          pretraining for NLP tasks.
        - Paper/report: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf?utm_source=chatgpt.com

    Example:
        >>> import torch
        >>> from stackformer.models.OpenAI import GPT_1
        >>> model = GPT_1(
        ...     vocab_size=50257,
        ...     num_layers=4,
        ...     embed_dim=512,
        ...     num_heads=8,
        ...     seq_len=128,
        ...     dropout=0.1,
        ...     hidden_dim=2048,
        ... )
        >>> x = torch.randint(0, 50257, (2, 32))
        >>> logits = model(x)
        >>> out = model.generate(x, max_new_tokens=16)
    """
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

        self.final_norm = nn.LayerNorm(embed_dim, eps=eps, device=device, dtype=dtype)
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

        self.norm1 = nn.LayerNorm(embed_dim, eps=eps, device=device, dtype=dtype)
        self.ff = FF_GELU(embed_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(embed_dim, eps=eps, device=device, dtype=dtype)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# --- GPT-2 Decoder ---
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
    """GPT-2 style decoder-only causal language model.

    Simple explanation:
        GPT-2 scales decoder-only transformers and uses pre-normalization for
        better optimization behavior in deeper stacks.

    Architecture details:
        - Attention: Multi-Head self-attention.
        - Masking: Causal mask.
        - Position encoding: Learned absolute positional embeddings.
        - Feed-forward: GELU MLP.
        - Normalization: Pre-LayerNorm in blocks + final LayerNorm.

    Research context:
        - Historical role: demonstrated strong zero/few-shot generation by
          scaling model and data.
        - Paper/report: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf?utm_source=chatgpt.com

    Example:
        >>> import torch
        >>> from stackformer.models.OpenAI import GPT_2
        >>> model = GPT_2(
        ...     vocab_size=50257,
        ...     num_layers=6,
        ...     embed_dim=768,
        ...     num_heads=12,
        ...     seq_len=128,
        ...     dropout=0.1,
        ...     hidden_dim=3072,
        ... )
        >>> x = torch.randint(0, 50257, (1, 24))
        >>> logits = model(x)
        >>> out = model.generate(x, max_new_tokens=20)
    """
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

        self.final_norm = nn.LayerNorm(embed_dim, eps=eps, device=device, dtype=dtype)
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
