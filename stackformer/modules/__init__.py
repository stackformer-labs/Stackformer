"""Public module-level API for reusable Stackformer building blocks."""

from .Attention import (
    Cross_MultiHead_Attention,
    Group_query_Attention,
    Group_query_Attention_With_RoPE,
    Multi_Head_Attention,
    Multi_Head_Attention_With_RoPE,
    Multi_query_Attention,
    Multi_query_Attention_With_RoPE,
    Self_Attention,
    kv_cache_group_query,
    kv_cache_multihead,
)
from .Feed_forward import (
    FF_GELU,
    FF_GeGLU,
    FF_LeakyReLU,
    FF_ReLU,
    FF_Sigmoid,
    FF_SiLU,
    FF_SwiGLU,
)
from .Masking import make_mask
from .Normalization import LayerNormalization, RMSNormalization
from .position_embedding import (
    AbsolutePositionEmbedding,
    RoPE,
    SinusoidalPositionalEmbedding,
)

TokenizerImportError: Exception | None = None

try:
    from .tokenizer import Embedding_using_tiktoken
except Exception as exc:  # optional dependency: tiktoken
    TokenizerImportError = exc
    Embedding_using_tiktoken = None  # type: ignore[assignment]

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
    "Group_query_Attention",
    "Group_query_Attention_With_RoPE",
    "LayerNormalization",
    "Multi_Head_Attention",
    "Multi_Head_Attention_With_RoPE",
    "Multi_query_Attention",
    "Multi_query_Attention_With_RoPE",
    "RMSNormalization",
    "RoPE",
    "Self_Attention",
    "SinusoidalPositionalEmbedding",
    "kv_cache_group_query",
    "kv_cache_multihead",
    "make_mask",
]

if Embedding_using_tiktoken is not None:
    __all__.append("Embedding_using_tiktoken")
