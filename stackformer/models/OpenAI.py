"""OpenAI GPT-family decoder-only model implementations.

This module contains GPT-1 and GPT-2 style language models rebuilt on top of
the Stackformer block builders (BlockConfig + TransformerEncoder).

Key architecture difference between GPT-1 and GPT-2:
    GPT-1 → Post-LayerNorm  (pre_norm=False, norm="layernorm")
    GPT-2 → Pre-LayerNorm   (pre_norm=True,  norm="layernorm")
"""

import torch
import torch.nn as nn

from stackformer.modules.layer import BlockConfig, TransformerEncoder
from stackformer.modules.position_embedding import AbsolutePositionEmbedding
from stackformer.generate import text_generate


# ─────────────────────────────────────────────────────────────────────────────
# GPT-1
# ─────────────────────────────────────────────────────────────────────────────

class GPT_1(nn.Module):
    """GPT-1 style decoder-only causal language model.

    Simple explanation:
        GPT-1 is an early large-scale autoregressive transformer that predicts
        the next token from previous tokens. It uses Post-LayerNorm, learned
        absolute position embeddings, and GELU feed-forward blocks.

    Architecture details:
        - Attention:          Multi-Head Attention (MHA), causal mask.
        - Position encoding:  Learned absolute positional embeddings.
        - Feed-forward:       GELU MLP.
        - Normalization:      Post-LayerNorm  (pre_norm=False).
        - QKV bias:           True  (original GPT-1 convention).

    Research context:
        - Historical role: established transfer learning with generative
          pretraining for NLP tasks.
        - Paper: https://cdn.openai.com/research-covers/language-unsupervised/
                 language_understanding_paper.pdf

    Example::

        import torch
        from stackformer.models.OpenAI import GPT_1

        model = GPT_1(
            vocab_size=50257,
            num_layers=4,
            embed_dim=512,
            num_heads=8,
            seq_len=128,
            dropout=0.1,
            hidden_dim=2048,
        )
        x      = torch.randint(0, 50257, (2, 32))
        logits = model(x)                          # (2, 32, 50257)
        out    = model.generate(x, max_new_tokens=16)
    """

    def __init__(
        self,
        vocab_size:  int,
        num_layers:  int,
        embed_dim:   int,
        num_heads:   int,
        seq_len:     int,
        dropout:     float         = 0.1,
        hidden_dim:  int           = 0,       # 0 → 4 × embed_dim
        qkv_bias:    bool          = True,    # GPT-1 uses bias
        eps:         float         = 1e-5,
        device:      str           = "cpu",
        dtype:       torch.dtype   = torch.float32,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.seq_len = seq_len

        cfg = BlockConfig(
            embed_dim  = embed_dim,
            num_heads  = num_heads,
            hidden_dim = hidden_dim,
            attention  = "mha",
            ffn        = "gelu",
            norm       = "layernorm",
            pre_norm   = False,       # ← Post-LN: GPT-1 original
            dropout    = dropout,
            qkv_bias   = qkv_bias,
            device     = device,
            dtype      = dtype,
        )

        self.embedding          = nn.Embedding(vocab_size, embed_dim, device=device, dtype=dtype)
        self.position_embedding = AbsolutePositionEmbedding(seq_len, embed_dim, device=device, dtype=dtype)

        # pos_embedding=None because we handle it in forward (token + position)
        self.backbone = TransformerEncoder(cfg, num_layers=num_layers, pos_embedding=None)

        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) + self.position_embedding(x)
        x = self.backbone(x, mask=True)   # causal mask on — decoder-only
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return text_generate(self, *args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# GPT-2
# ─────────────────────────────────────────────────────────────────────────────

class GPT_2(nn.Module):
    """GPT-2 style decoder-only causal language model.

    Simple explanation:
        GPT-2 scales the decoder-only transformer and switches to Pre-LayerNorm
        for better gradient flow in deeper stacks. Otherwise the architecture
        is identical to GPT-1.

    Architecture details:
        - Attention:          Multi-Head Attention (MHA), causal mask.
        - Position encoding:  Learned absolute positional embeddings.
        - Feed-forward:       GELU MLP.
        - Normalization:      Pre-LayerNorm  (pre_norm=True)  +  final norm.
        - QKV bias:           True  (GPT-2 convention).

    Research context:
        - Historical role: demonstrated strong zero/few-shot generation by
          scaling model and data.
        - Paper: https://cdn.openai.com/better-language-models/
                 language_models_are_unsupervised_multitask_learners.pdf

    Example::

        import torch
        from stackformer.models.OpenAI import GPT_2

        model = GPT_2(
            vocab_size=50257,
            num_layers=6,
            embed_dim=768,
            num_heads=12,
            seq_len=128,
            dropout=0.1,
            hidden_dim=3072,
        )
        x      = torch.randint(0, 50257, (1, 24))
        logits = model(x)                          # (1, 24, 50257)
        out    = model.generate(x, max_new_tokens=20)
    """

    def __init__(
        self,
        vocab_size:  int,
        num_layers:  int,
        embed_dim:   int,
        num_heads:   int,
        seq_len:     int,
        dropout:     float         = 0.1,
        hidden_dim:  int           = 0,       # 0 → 4 × embed_dim
        qkv_bias:    bool          = True,    # GPT-2 uses bias
        eps:         float         = 1e-5,
        device:      str           = "cpu",
        dtype:       torch.dtype   = torch.float32,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.seq_len = seq_len

        cfg = BlockConfig(
            embed_dim  = embed_dim,
            num_heads  = num_heads,
            hidden_dim = hidden_dim,
            attention  = "mha",
            ffn        = "gelu",
            norm       = "layernorm",
            pre_norm   = True,        # ← Pre-LN: GPT-2 difference
            dropout    = dropout,
            qkv_bias   = qkv_bias,
            device     = device,
            dtype      = dtype,
        )

        self.embedding          = nn.Embedding(vocab_size, embed_dim, device=device, dtype=dtype)
        self.position_embedding = AbsolutePositionEmbedding(seq_len, embed_dim, device=device, dtype=dtype)

        # TransformerEncoder adds final_norm automatically for pre_norm=True
        self.backbone = TransformerEncoder(cfg, num_layers=num_layers, pos_embedding=None)

        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) + self.position_embedding(x)
        x = self.backbone(x, mask=True)   # causal mask on — decoder-only
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return text_generate(self, *args, **kwargs)