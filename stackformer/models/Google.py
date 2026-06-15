"""Google-family decoder-only model implementations.

This module includes two Gemma-style causal language models rebuilt on top of
the Stackformer block builders (BlockConfig + TransformerEncoder):

    gemma_1_2b  —  Multi-Query Attention + RoPE  (MQA, lighter KV)
    gemma_1_7b  —  Multi-Head Attention  + RoPE  (MHA, full capacity)

Both use GeGLU feed-forward and RMSNorm with pre-normalization.
Paper: https://arxiv.org/pdf/2403.08295
"""

import torch
import torch.nn as nn

from stackformer.modules.layer import BlockConfig, TransformerEncoder
from stackformer.generate import text_generate


# ─────────────────────────────────────────────────────────────────────────────
# Gemma 1 2B
# ─────────────────────────────────────────────────────────────────────────────

class gemma_1_2b(nn.Module):
    """Gemma 1 2B-style decoder-only causal language model.

    Simple explanation:
        A GPT-like text generation model that predicts the next token from left
        to right. Uses Multi-Query Attention to reduce the KV cache footprint
        at inference time while keeping a full set of query heads.

    Architecture details:
        - Attention:         Multi-Query Attention (MQA) with RoPE.
        - Masking:           Causal mask.
        - Position encoding: RoPE (built into the attention module).
        - Feed-forward:      GeGLU.
        - Normalization:     Pre-norm RMSNorm in every block + final RMSNorm.

    Research context:
        - Family: Gemma-style decoder language models.
        - Why MQA: one shared KV head dramatically reduces KV cache size and
          memory bandwidth during autoregressive generation.
        - Paper: https://arxiv.org/pdf/2403.08295

    Example::

        import torch
        from stackformer.models.Google import gemma_1_2b

        model = gemma_1_2b(
            vocab_size=32000,
            num_layers=4,
            embed_dim=512,
            num_heads=8,
            seq_len=128,
            dropout=0.1,
            hidden_dim=2048,
        )
        input_ids = torch.randint(0, 32000, (2, 64))
        logits    = model(input_ids)                    # (2, 64, 32000)
        generated = model.generate(input_ids, max_new_tokens=16)
    """

    def __init__(
        self,
        vocab_size:  int,
        num_layers:  int,
        embed_dim:   int,
        num_heads:   int,
        seq_len:     int,
        dropout:     float       = 0.0,
        hidden_dim:  int         = 0,     # 0 → 4 × embed_dim
        eps:         float       = 1e-5,
        device:      str         = "cpu",
        dtype:       torch.dtype = torch.float32,
    ):
        super().__init__()
        self.seq_len = seq_len

        cfg = BlockConfig(
            embed_dim  = embed_dim,
            num_heads  = num_heads,
            hidden_dim = hidden_dim,
            attention  = "mqa_rope",  # ← MQA + RoPE: Gemma 2B variant
            ffn        = "geglu",
            norm       = "rmsnorm",
            pre_norm   = True,
            dropout    = dropout,
            device     = device,
            dtype      = dtype,
        )

        self.embedding = nn.Embedding(vocab_size, embed_dim, dtype=dtype, device=device)

        # pos_embedding=None because RoPE is built into mqa_rope attention
        self.backbone = TransformerEncoder(cfg, num_layers=num_layers, pos_embedding=None)

        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) * (self.embedding.embedding_dim ** 0.5)  # scale factor for stability
        x = self.backbone(x, mask=True) # causal mask
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt_ids, max_context_len=128, max_new_tokens=50,
                 temperature=1.0, top_k=None, top_p=1.0, eos_token_id=None):
        return text_generate(self, prompt_ids, max_context_len, max_new_tokens,
                             temperature, top_k, top_p, eos_token_id)


# ─────────────────────────────────────────────────────────────────────────────
# Gemma 1 7B
# ─────────────────────────────────────────────────────────────────────────────

class gemma_1_7b(nn.Module):
    """Gemma 1 7B-style decoder-only causal language model.

    Simple explanation:
        Larger Gemma-style model with full Multi-Head Attention. Every head has
        its own K and V projections, giving more expressive power than MQA at
        the cost of a larger KV cache.

    Architecture details:
        - Attention:         Multi-Head Attention (MHA) with RoPE.
        - Masking:           Causal mask.
        - Position encoding: RoPE (built into the attention module).
        - Feed-forward:      GeGLU.
        - Normalization:     Pre-norm RMSNorm in every block + final RMSNorm.

    Research context:
        - Family: Gemma-style decoder language models.
        - Tradeoff: higher modeling capacity than the 2B MQA variant at
          increased memory and compute cost.
        - Paper: https://arxiv.org/pdf/2403.08295

    Example::

        import torch
        from stackformer.models.Google import gemma_1_7b

        model = gemma_1_7b(
            vocab_size=32000,
            num_layers=6,
            embed_dim=768,
            num_heads=12,
            seq_len=128,
            dropout=0.1,
            hidden_dim=3072,
        )
        input_ids = torch.randint(0, 32000, (1, 32))
        logits    = model(input_ids)
        generated = model.generate(input_ids, max_new_tokens=20)
    """

    def __init__(
        self,
        vocab_size:  int,
        num_layers:  int,
        embed_dim:   int,
        num_heads:   int,
        seq_len:     int,
        dropout:     float       = 0.0,
        hidden_dim:  int         = 0,     # 0 → 4 × embed_dim
        eps:         float       = 1e-5,
        device:      str         = "cpu",
        dtype:       torch.dtype = torch.float32,
    ):
        super().__init__()
        self.seq_len = seq_len

        cfg = BlockConfig(
            embed_dim  = embed_dim,
            num_heads  = num_heads,
            hidden_dim = hidden_dim,
            attention  = "mha_rope",  # ← MHA + RoPE: Gemma 7B variant
            ffn        = "geglu",
            norm       = "rmsnorm",
            pre_norm   = True,
            dropout    = dropout,
            device     = device,
            dtype      = dtype,
        )

        self.embedding = nn.Embedding(vocab_size, embed_dim, dtype=dtype, device=device)

        # pos_embedding=None because RoPE is built into mha_rope attention
        self.backbone = TransformerEncoder(cfg, num_layers=num_layers, pos_embedding=None)

        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, dtype=dtype, device=device)
    
    # in forward(), both models
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) * (self.embedding.embedding_dim ** 0.5)  # scale factor for stability
        x = self.backbone(x, mask=True) # causal mask
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt_ids, max_context_len=128, max_new_tokens=50,
                 temperature=1.0, top_k=None, top_p=1.0, eos_token_id=None):
        return text_generate(self, prompt_ids, max_context_len, max_new_tokens,
                             temperature, top_k, top_p, eos_token_id)