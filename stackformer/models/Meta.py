"""Meta-family decoder-only model implementations.

This module provides LLaMA-style causal language models rebuilt on top of
the Stackformer block builders (BlockConfig + TransformerEncoder):

    llama_1  —  MHA + RoPE + SwiGLU  (standard training-time model)
    llama_2  —  GQA + KV cache + RoPE + SwiGLU  (inference-optimised)

Note on llama_2:
    LLaMA 2 uses a stateful KV cache (kv_cache_group_query) whose forward
    signature requires a start_pos argument. The KV cache block is kept
    separate from the standard TransformerEncoder stack — it is not a
    drop-in replacement — so llama_2 builds its own block/decoder layer
    while still using BlockConfig to share the FFN and norm configuration.

Paper: https://arxiv.org/abs/2302.13971  (LLaMA 1)
       https://arxiv.org/abs/2307.09288  (LLaMA 2)
"""

import torch
import torch.nn as nn

from stackformer.modules.layer import BlockConfig, TransformerEncoder, _build_ffn, _build_norm
from stackformer.modules.Attention import kv_cache_group_query
from stackformer.generate import text_generate


# ─────────────────────────────────────────────────────────────────────────────
# LLaMA 1
# ─────────────────────────────────────────────────────────────────────────────

class llama_1(nn.Module):
    """LLaMA 1-style decoder-only causal language model.

    Simple explanation:
        Reads token sequences and predicts the next token using only left
        context (causal decoding). Designed for autoregressive text generation
        with RoPE position encoding and SwiGLU feed-forward blocks.

    Architecture details:
        - Attention:         Multi-Head Attention (MHA) with RoPE.
        - Masking:           Causal mask.
        - Position encoding: RoPE (built into the attention module).
        - Feed-forward:      SwiGLU.
        - Normalization:     Pre-norm RMSNorm in every block + final RMSNorm.

    Research context:
        - Family: LLaMA 1 generation of decoder transformer models.
        - Typical use: efficient high-quality language modeling and generation.
        - Paper: https://arxiv.org/abs/2302.13971

    Example::

        import torch
        from stackformer.models.Meta import llama_1

        model = llama_1(
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
        generated = model.generate(input_ids, max_new_tokens=32)
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
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.seq_len = seq_len

        cfg = BlockConfig(
            embed_dim  = embed_dim,
            num_heads  = num_heads,
            hidden_dim = hidden_dim,
            attention  = "mha_rope",  # ← MHA + RoPE: LLaMA 1
            ffn        = "swiglu",
            norm       = "rmsnorm",
            pre_norm   = True,
            dropout    = dropout,
            device     = device,
            dtype      = dtype,
        )

        self.embedding = nn.Embedding(vocab_size, embed_dim, dtype=dtype, device=device)

        # pos_embedding=None — RoPE is handled inside mha_rope
        self.backbone = TransformerEncoder(cfg, num_layers=num_layers, pos_embedding=None)

        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.backbone(x, mask=True)   # causal mask
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt_ids, max_context_len=128, max_new_tokens=50,
                 temperature=1.0, top_k=None, top_p=1.0, eos_token_id=None):
        return text_generate(self, prompt_ids, max_context_len, max_new_tokens,
                             temperature, top_k, top_p, eos_token_id)


# ─────────────────────────────────────────────────────────────────────────────
# LLaMA 2  (stateful KV cache — needs its own block wiring)
# ─────────────────────────────────────────────────────────────────────────────

class _llama_2_Block(nn.Module):
    """One LLaMA 2 decoder block with GQA + persistent KV cache.

    Uses BlockConfig to stay consistent with the rest of the stack for FFN
    and norm choices. The attention is kv_cache_group_query, which has a
    stateful cache and requires start_pos in forward — it cannot be swapped
    out via the standard TransformerEncoder stack.
    """

    def __init__(self, cfg: BlockConfig, batch_size: int, kv_seq_len: int):
        super().__init__()
        self.norm1 = _build_norm(cfg)
        self.attn  = kv_cache_group_query(
            embed_dim        = cfg.embed_dim,
            num_query_heads  = cfg.num_heads,
            num_kv_heads     = cfg.num_kv_heads,
            batch_size       = batch_size,
            kv_seq_len       = kv_seq_len,
            dropout          = cfg.dropout,
            device           = cfg.device,
            dtype            = cfg.dtype,
        )
        self.norm2 = _build_norm(cfg)
        self.ffn   = _build_ffn(cfg)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), start_pos, rope=True)
        x = x + self.ffn(self.norm2(x))
        return x


class llama_2(nn.Module):
    """LLaMA 2-style decoder-only causal language model with GQA + KV cache.

    Simple explanation:
        Autoregressive decoder that supports grouped-query attention with a
        persistent key/value cache. The cache makes generation faster in long
        decoding loops because past K/V states are stored and reused instead
        of being recomputed every step.

    Architecture details:
        - Attention:         Grouped-Query Attention (GQA) with KV cache.
        - Masking:           Causal mask (handled inside kv_cache_group_query).
        - Position encoding: RoPE (applied at each attention call).
        - Feed-forward:      SwiGLU.
        - Normalization:     Pre-norm RMSNorm in every block + final RMSNorm.

    Args:
        num_query_heads: Total number of query heads.
        num_kv_heads:    Number of KV heads  (num_query_heads % num_kv_heads == 0).
        batch_size:      Cache pre-allocation size. Must match inference batch size.
        kv_seq_len:      Maximum cache sequence length (doubled internally to avoid OOB).

    Research context:
        - Family: LLaMA 2 generation decoder transformers.
        - GQA: num_query_heads heads share num_kv_heads KV projections, shrinking
          the cache by a factor of (num_query_heads / num_kv_heads).
        - Paper: https://arxiv.org/abs/2307.09288

    Example::

        import torch
        from stackformer.models.Meta import llama_2

        model = llama_2(
            vocab_size=32000,
            num_layers=4,
            embed_dim=512,
            num_query_heads=8,
            num_kv_heads=2,
            batch_size=1,
            kv_seq_len=128,
            hidden_dim=2048,
        )
        input_ids = torch.randint(0, 32000, (1, 16))
        logits    = model(input_ids, start_pos=0)
        generated = model.generate(input_ids, max_new_tokens=24)
    """

    def __init__(
        self,
        vocab_size:       int,
        num_layers:       int,
        embed_dim:        int,
        num_query_heads:  int,
        num_kv_heads:     int,
        batch_size:       int,
        kv_seq_len:       int,
        hidden_dim:       int         = 0,     # 0 → 4 × embed_dim
        dropout:          float       = 0.0,
        eps:              float       = 1e-5,
        dtype:            torch.dtype = torch.float32,
        device:           str         = "cpu",
    ):
        super().__init__()
        assert embed_dim % num_query_heads == 0, "embed_dim must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        self.seq_len = kv_seq_len
        
        # in llama_2.__init__(), before BlockConfig
        # if hidden_dim == 0:
        #     'hidden_dim = int(int(8 / 3 * embed_dim) / 256) * 256 # 8/3 is the LLaMA 2 expansion ratio, and we round to a multiple of 256 for efficiency
    
        # BlockConfig drives FFN and norm — attention is set separately    
        cfg = BlockConfig(
            embed_dim    = embed_dim,
            num_heads    = num_query_heads,
            num_kv_heads = num_kv_heads,
            hidden_dim   = hidden_dim,
            attention    = "gqa_rope",   # informational — overridden by kv_cache block
            ffn          = "swiglu",
            norm         = "rmsnorm",
            pre_norm     = True,
            dropout      = dropout,
            device       = device,
            dtype        = dtype,
        )

        self.embedding = nn.Embedding(vocab_size, embed_dim, dtype=dtype, device=device)

        self.layers = nn.ModuleList([
            _llama_2_Block(cfg, batch_size=batch_size, kv_seq_len=kv_seq_len)
            for _ in range(num_layers)
        ])

        self.final_norm = _build_norm(cfg)
        self.lm_head    = nn.Linear(embed_dim, vocab_size, bias=False, dtype=dtype, device=device)

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, start_pos)
        x = self.final_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt_ids, max_context_len=128, max_new_tokens=50,
                 temperature=1.0, top_k=None, top_p=1.0, eos_token_id=None):
        return text_generate(self, prompt_ids, max_context_len, max_new_tokens,
                             temperature, top_k, top_p, eos_token_id)
