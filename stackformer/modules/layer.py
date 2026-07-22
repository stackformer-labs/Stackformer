"""stackformer/modules/layer.py

Transformer building blocks — from a single block to a full encoder/decoder stack.

Design principles:
  - One config object (BlockConfig) drives everything — no repeated keyword walls.
  - Private factory functions (_build_*) keep each block class clean.
  - EncoderBlock / DecoderBlock own their wiring; stacks own their depth.
  - TransformerBlock is kept for backward compatibility.

Typical usage::

    from stackformer.modules.layer import BlockConfig, TransformerEncoder, TransformerDecoder

    cfg = BlockConfig(embed_dim=512, num_heads=8, ffn="swiglu", attention="gqa", num_kv_heads=2)

    encoder = TransformerEncoder(cfg, num_layers=6, pos_embedding="sinusoidal")
    decoder = TransformerDecoder(cfg, num_layers=6)

    memory = encoder(src)           # (B, S, C)
    out    = decoder(tgt, memory)   # (B, T, C)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

from stackformer.modules.Attention import (
    Cross_MultiHead_Attention,
    Group_query_Attention,
    Group_query_Attention_With_RoPE,
    Multi_Head_Attention,
    Multi_Head_Attention_With_RoPE,
    Multi_query_Attention,
    Multi_query_Attention_With_RoPE,
    Self_Attention,
)
from stackformer.modules.Feed_forward import (
    FF_GELU, FF_GeGLU, FF_LeakyReLU,
    FF_ReLU, FF_Sigmoid, FF_SiLU, FF_SwiGLU,
)
from stackformer.modules.Normalization import (
    LayerNormalization, RMSNormalization,
)
from stackformer.modules.position_embedding import (
    AbsolutePositionEmbedding,
    SinusoidalPositionalEmbedding,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BlockConfig:
    """All hyperparameters that describe one transformer block.

    Pass a single ``BlockConfig`` to any block or stack class instead of
    a wall of keyword arguments.  Every field has a sensible default so
    you only need to specify what differs from the baseline.

    Args:
        embed_dim:    Model width ``C``.  Required.
        num_heads:    Number of query heads ``H``.  Required.
        hidden_dim:   FFN inner dimension.  Defaults to ``4 * embed_dim``.
        attention:    Self-attention variant.  One of:
                      ``'mha'``, ``'mha_rope'``,
                      ``'mqa'``, ``'mqa_rope'``,
                      ``'gqa'``, ``'gqa_rope'``,
                      ``'self'``.
        num_kv_heads: KV heads for GQA variants.  Defaults to ``num_heads``
                      (equivalent to standard MHA).
        ffn:          Feed-forward activation.  One of:
                      ``'relu'``, ``'leakyrelu'``, ``'gelu'``,
                      ``'sigmoid'``, ``'silu'``, ``'swiglu'``, ``'geglu'``.
        norm:         Normalization layer.  One of:
                      ``'layernorm'``, ``'rmsnorm'``.
        dropout:      Dropout probability applied inside attention and FFN.
        pre_norm:     ``True``  → Pre-LN (norm before sub-layer, stable).
                      ``False`` → Post-LN (norm after residual, original paper).
        mask_type:    List of mask pattern names forwarded to each attention
                      module (e.g. ``['causal']``, ``['sliding_window']``).
        qkv_bias:     Whether Q/K/V projection layers have bias terms.
        device:       Compute and parameter device.
        dtype:        Parameter dtype.

    Examples::

        # Minimal — sensible defaults for everything else
        cfg = BlockConfig(embed_dim=512, num_heads=8)

        # Modern LLM style
        cfg = BlockConfig(
            embed_dim=4096,
            num_heads=32,
            ffn="swiglu",
            attention="gqa",
            num_kv_heads=8,
            norm="rmsnorm",
        )

        # Small debug config
        cfg = BlockConfig(embed_dim=64, num_heads=4, hidden_dim=128, dropout=0.1)
    """

    embed_dim:    int
    num_heads:    int
    hidden_dim:   int = 0            # 0 → auto: 4 × embed_dim

    attention:    Literal[
        "mha", "mha_rope",
        "mqa", "mqa_rope",
        "gqa", "gqa_rope",
        "self",
    ] = "mha"
    num_kv_heads: int | None = None  # None → same as num_heads

    ffn: Literal[
        "relu", "leakyrelu", "gelu",
        "sigmoid", "silu", "swiglu", "geglu",
    ] = "swiglu"
    norm:    Literal["layernorm", "rmsnorm"] = "rmsnorm"

    dropout:  float = 0.0
    pre_norm: bool  = True

    mask_type: list = field(default_factory=lambda: ["causal"])
    qkv_bias:  bool = False

    device: str            = "cpu"
    dtype:  torch.dtype    = torch.float32

    def __post_init__(self) -> None:
        if self.hidden_dim == 0:
            self.hidden_dim = 4 * self.embed_dim

        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        if "gqa" in self.attention and self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads}) for GQA"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Private factories  (one concern each — easy to extend)
# ─────────────────────────────────────────────────────────────────────────────

def _build_attention(cfg: BlockConfig) -> nn.Module:
    """Return the self-attention module described by *cfg*."""
    shared = dict(
        dropout   = cfg.dropout,
        mask_type = cfg.mask_type,
        qkv_bias  = cfg.qkv_bias,
        device    = cfg.device,
        dtype     = cfg.dtype,
    )
    E, H, G = cfg.embed_dim, cfg.num_heads, cfg.num_kv_heads

    registry: dict[str, nn.Module] = {
        "mha":      Multi_Head_Attention(E, H, **shared),
        "mha_rope": Multi_Head_Attention_With_RoPE(E, H, **shared),
        "mqa":      Multi_query_Attention(E, H, **shared),
        "mqa_rope": Multi_query_Attention_With_RoPE(E, H, **shared),
        "gqa":      Group_query_Attention(E, H, G, **shared),
        "gqa_rope": Group_query_Attention_With_RoPE(E, H, G, **shared),
        "self":     Self_Attention(E, **shared),
    }
    key = cfg.attention.lower()
    if key not in registry:
        raise ValueError(
            f"Unknown attention '{cfg.attention}'. "
            f"Available: {list(registry)}"
        )
    return registry[key]


def _build_cross_attention(cfg: BlockConfig) -> nn.Module:
    """Return a cross-attention module (queries from target, K/V from memory)."""
    return Cross_MultiHead_Attention(
        cfg.embed_dim, cfg.num_heads,
        dropout  = cfg.dropout,
        qkv_bias = cfg.qkv_bias,
        device   = cfg.device,
        dtype    = cfg.dtype,
    )


def _build_ffn(cfg: BlockConfig) -> nn.Module:
    """Return the FFN module described by *cfg*."""
    hw = dict(device=cfg.device, dtype=cfg.dtype)
    E, M, D = cfg.embed_dim, cfg.hidden_dim, cfg.dropout

    registry: dict[str, nn.Module] = {
        "relu":      FF_ReLU(E, M, D, **hw),
        "leakyrelu": FF_LeakyReLU(E, M, D, **hw),
        "gelu":      FF_GELU(E, M, D, **hw),
        "sigmoid":   FF_Sigmoid(E, M, D, **hw),
        "silu":      FF_SiLU(E, M, D, **hw),
        "swiglu":    FF_SwiGLU(E, M, D, **hw),
        "geglu":     FF_GeGLU(E, M, D, **hw),
    }
    key = cfg.ffn.lower()
    if key not in registry:
        raise ValueError(
            f"Unknown FFN '{cfg.ffn}'. "
            f"Available: {list(registry)}"
        )
    return registry[key]


def _build_norm(cfg: BlockConfig) -> nn.Module:
    """Return one normalization layer described by *cfg*."""
    hw = dict(device=cfg.device, dtype=cfg.dtype)

    registry: dict[str, nn.Module] = {
        "layernorm": LayerNormalization(cfg.embed_dim, **hw),
        "rmsnorm":   RMSNormalization(cfg.embed_dim, **hw),
    }
    key = cfg.norm.lower()
    if key not in registry:
        raise ValueError(
            f"Unknown norm '{cfg.norm}'. "
            f"Available: {list(registry)}"
        )
    return registry[key]


def _build_pos_embedding(
    name: str | None,
    cfg: BlockConfig,
    max_seq_len: int,
) -> nn.Module | None:
    """Return a positional embedding module, or None.

    Note: RoPE positional encoding is handled *inside* the attention
    modules when using ``attention='mha_rope'`` / ``'gqa_rope'`` etc.
    Do not add a separate pos_embedding in that case.
    """
    if name is None:
        return None

    hw = dict(device=cfg.device, dtype=cfg.dtype)

    registry: dict[str, nn.Module] = {
        "sinusoidal": SinusoidalPositionalEmbedding(max_seq_len, cfg.embed_dim, **hw),
        "absolute":   AbsolutePositionEmbedding(max_seq_len, cfg.embed_dim, **hw),
    }
    key = name.lower()
    if key not in registry:
        raise ValueError(
            f"Unknown positional embedding '{name}'. "
            f"Available: {list(registry)}"
        )
    return registry[key]


# ─────────────────────────────────────────────────────────────────────────────
# Blocks
# ─────────────────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """One transformer encoder layer: self-attention + FFN.

    Wiring (pre-norm)::

        x → norm1 → self_attn ─┐
        └──────────────────────⊕ → norm2 → ffn ─┐
                                └────────────────⊕ → output

    Wiring (post-norm)::

        x → self_attn ─┐
        └──────────────⊕ → norm1 → ffn ─┐
                                         ⊕ → norm2 → output

    Args:
        cfg: :class:`BlockConfig` that specifies all sub-modules.

    Forward args:
        x:    ``(B, T, C)`` input tensor.
        mask: Whether to apply the configured attention mask.
              Default ``False`` — encoders typically use bidirectional attention.

    Returns:
        ``(B, T, C)``

    Example::

        cfg   = BlockConfig(embed_dim=512, num_heads=8, ffn="swiglu")
        block = EncoderBlock(cfg)
        y     = block(torch.randn(2, 64, 512))
    """

    def __init__(self, cfg: BlockConfig) -> None:
        super().__init__()
        self.pre_norm  = cfg.pre_norm
        self.self_attn = _build_attention(cfg)
        self.ffn       = _build_ffn(cfg)
        self.norm1     = _build_norm(cfg)
        self.norm2     = _build_norm(cfg)

    def forward(self, x: torch.Tensor, mask: bool = False) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.self_attn(self.norm1(x), mask=mask)
            x = x + self.ffn(self.norm2(x))
        else:   # post-norm
            x = self.norm1(x + self.self_attn(x, mask=mask))
            x = self.norm2(x + self.ffn(x))
        return x


class DecoderBlock(nn.Module):
    """One transformer decoder layer: causal self-attn + cross-attn + FFN.

    Wiring (pre-norm)::

        x → norm1 → self_attn (causal) ─┐
        └────────────────────────────────⊕
          → norm2 → cross_attn(memory) ──┐
          └────────────────────────────── ⊕
            → norm3 → ffn ───────────────┐
            └────────────────────────────⊕ → output

    The cross-attention always uses standard MHA (queries from *x*,
    keys/values from *memory*). The self-attention variant is taken from
    *cfg.attention* — so you can use GQA self-attention in the decoder.

    Args:
        cfg: :class:`BlockConfig` that specifies all sub-modules.

    Forward args:
        x:               ``(B, T, C)`` target sequence.
        memory:          ``(B, S, C)`` encoder output.
        self_mask:       Apply causal mask on self-attention.  Default ``True``.
        cross_mask:      Apply mask on cross-attention.  Default ``False``.
        cross_attn_mask: Optional explicit ``(T, S)`` boolean mask for
                         cross-attention (overrides ``cross_mask``).

    Returns:
        ``(B, T, C)``

    Example::

        cfg   = BlockConfig(embed_dim=512, num_heads=8)
        block = DecoderBlock(cfg)
        y     = block(tgt, memory)
    """

    def __init__(self, cfg: BlockConfig) -> None:
        super().__init__()
        self.pre_norm   = cfg.pre_norm
        self.self_attn  = _build_attention(cfg)
        self.cross_attn = _build_cross_attention(cfg)
        self.ffn        = _build_ffn(cfg)
        self.norm1      = _build_norm(cfg)
        self.norm2      = _build_norm(cfg)
        self.norm3      = _build_norm(cfg)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask:       bool                  = True,
        cross_mask:      bool                  = False,
        cross_attn_mask: torch.Tensor | None   = None,
    ) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.self_attn(self.norm1(x), mask=self_mask)
            x = x + self.cross_attn(self.norm2(x), memory,
                                    mask=cross_mask, attn_mask=cross_attn_mask)
            x = x + self.ffn(self.norm3(x))
        else:   # post-norm
            x = self.norm1(x + self.self_attn(x, mask=self_mask))
            x = self.norm2(x + self.cross_attn(x, memory,
                                                mask=cross_mask, attn_mask=cross_attn_mask))
            x = self.norm3(x + self.ffn(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Stacks
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """A stack of :class:`EncoderBlock` layers.

    Args:
        cfg:           :class:`BlockConfig` shared by every block.
        num_layers:    Number of encoder blocks to stack.
        pos_embedding: Optional positional embedding added before the first
                       block.  One of ``'sinusoidal'``, ``'absolute'``, or
                       ``None``.  Pass ``None`` when using ``*_rope``
                       attention (RoPE is handled inside each block).
        max_seq_len:   Maximum sequence length for learnable/sinusoidal PE.
                       Ignored when ``pos_embedding=None``.

    Forward args:
        x:    ``(B, T, C)`` already-embedded token tensor.
        mask: Forwarded to every block.  Default ``False``.

    Returns:
        ``(B, T, C)``

    Example::

        cfg     = BlockConfig(embed_dim=512, num_heads=8, ffn="swiglu", attention="gqa", num_kv_heads=2)
        encoder = TransformerEncoder(cfg, num_layers=6, pos_embedding="sinusoidal")
        memory  = encoder(src_embeds)
    """

    def __init__(
        self,
        cfg:           BlockConfig,
        num_layers:    int,
        pos_embedding: str | None = None,
        max_seq_len:   int        = 4096,
    ) -> None:
        super().__init__()
        self.pos_embedding = _build_pos_embedding(pos_embedding, cfg, max_seq_len)
        self.layers        = nn.ModuleList([EncoderBlock(cfg) for _ in range(num_layers)])
        # Pre-LN models need a final norm; post-LN already has one in each block.
        self.final_norm    = _build_norm(cfg) if cfg.pre_norm else nn.Identity()

    def forward(self, x: torch.Tensor, mask: bool = False) -> torch.Tensor:
        if self.pos_embedding is not None:
            x = x + self.pos_embedding(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.final_norm(x)


class TransformerDecoder(nn.Module):
    """A stack of :class:`DecoderBlock` layers.

    Args:
        cfg:           :class:`BlockConfig` shared by every block.
        num_layers:    Number of decoder blocks to stack.
        pos_embedding: Optional positional embedding.  Same rules as
                       :class:`TransformerEncoder`.
        max_seq_len:   Maximum sequence length for PE.

    Forward args:
        x:               ``(B, T, C)`` target embeddings.
        memory:          ``(B, S, C)`` encoder output.
        self_mask:       Causal mask on self-attention.  Default ``True``.
        cross_mask:      Mask on cross-attention.  Default ``False``.
        cross_attn_mask: Optional explicit ``(T, S)`` mask.

    Returns:
        ``(B, T, C)``

    Example::

        cfg     = BlockConfig(embed_dim=512, num_heads=8)
        decoder = TransformerDecoder(cfg, num_layers=6)
        out     = decoder(tgt_embeds, memory)
    """

    def __init__(
        self,
        cfg:           BlockConfig,
        num_layers:    int,
        pos_embedding: str | None = None,
        max_seq_len:   int        = 4096,
    ) -> None:
        super().__init__()
        self.pos_embedding = _build_pos_embedding(pos_embedding, cfg, max_seq_len)
        self.layers        = nn.ModuleList([DecoderBlock(cfg) for _ in range(num_layers)])
        self.final_norm    = _build_norm(cfg) if cfg.pre_norm else nn.Identity()

    def forward(
        self,
        x:               torch.Tensor,
        memory:          torch.Tensor,
        self_mask:       bool                = True,
        cross_mask:      bool                = False,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pos_embedding is not None:
            x = x + self.pos_embedding(x)
        for layer in self.layers:
            x = layer(
                x, memory,
                self_mask=self_mask,
                cross_mask=cross_mask,
                cross_attn_mask=cross_attn_mask,
            )
        return self.final_norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible TransformerBlock
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Single encoder block — kept for backward compatibility.

    Prefer :class:`EncoderBlock` for new code.  This class accepts the same
    keyword arguments as the original ``TransformerBlock`` and delegates to
    :class:`EncoderBlock` internally.

    Example (old API still works)::

        block = TransformerBlock(embed_dim=256, num_heads=4, hidden_dim=1024)
        y = block(x)

    Example (new API via BlockConfig)::

        cfg   = BlockConfig(embed_dim=256, num_heads=4)
        block = EncoderBlock(cfg)
        y     = block(x)
    """

    def __init__(
        self,
        embed_dim:    int,
        num_heads:    int,
        hidden_dim:   int,
        attention:    str          = "mha",
        num_kv_heads: int | None   = None,
        ffn:          str          = "relu",
        norm:         str          = "layernorm",
        dropout:      float        = 0.0,
        pre_norm:     bool         = True,
        device:       str          = "cpu",
        dtype:        torch.dtype  = torch.float32,
    ) -> None:
        super().__init__()
        cfg = BlockConfig(
            embed_dim    = embed_dim,
            num_heads    = num_heads,
            hidden_dim   = hidden_dim,
            attention    = attention,
            num_kv_heads = num_kv_heads,
            ffn          = ffn,
            norm         = norm,
            dropout      = dropout,
            pre_norm     = pre_norm,
            device       = device,
            dtype        = dtype,
        )
        self._block = EncoderBlock(cfg)

    def forward(self, x: torch.Tensor, mask: bool = True) -> torch.Tensor:
        return self._block(x, mask=mask)