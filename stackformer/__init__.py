"""Top-level public API for Stackformer."""

from .generate import text_generate
from .models import GPT_1, GPT_2, gemma_1_2b, gemma_1_7b, llama_1, llama_2, transformer
from .modules import (
    AbsolutePositionEmbedding,
    Cross_MultiHead_Attention,
    FF_GELU,
    FF_GeGLU,
    FF_LeakyReLU,
    FF_ReLU,
    FF_Sigmoid,
    FF_SiLU,
    FF_SwiGLU,
    Group_query_Attention,
    Group_query_Attention_With_RoPE,
    LayerNormalization,
    Multi_Head_Attention,
    Multi_Head_Attention_With_RoPE,
    Multi_query_Attention,
    Multi_query_Attention_With_RoPE,
    RMSNormalization,
    RoPE,
    Self_Attention,
    SinusoidalPositionalEmbedding,
    kv_cache_group_query,
    kv_cache_multihead,
    make_mask,
)
from .vision import SegFormerB0, ViT

TokenizerImportError: Exception | None = None
TrainerImportError: Exception | None = None

try:
    from .engine import Trainer
except Exception as exc:
    TrainerImportError = exc
    Trainer = None  # type: ignore[assignment]

__all__ = [
    "AbsolutePositionEmbedding",
    "Cross_MultiHead_Attention",
    "FF_GELU",
    "FF_GeGLU",
    "FF_LeakyReLU",
    "FF_ReLU",
    "FF_Sigmoid",
    "FF_SiLU",
    "FF_SwiGLU",
    "GPT_1",
    "GPT_2",
    "Group_query_Attention",
    "Group_query_Attention_With_RoPE",
    "LayerNormalization",
    "Multi_Head_Attention",
    "Multi_Head_Attention_With_RoPE",
    "Multi_query_Attention",
    "Multi_query_Attention_With_RoPE",
    "RMSNormalization",
    "RoPE",
    "SegFormerB0",
    "Self_Attention",
    "SinusoidalPositionalEmbedding",
    "ViT",
    "gemma_1_2b",
    "gemma_1_7b",
    "kv_cache_group_query",
    "kv_cache_multihead",
    "llama_1",
    "llama_2",
    "make_mask",
    "text_generate",
    "transformer",
]

if Trainer is not None:
    __all__.append("Trainer")
