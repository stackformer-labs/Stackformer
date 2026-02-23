"""Public API for Stackformer.

This module intentionally uses lazy/guarded imports for optional dependencies so
`import stackformer` works in minimal environments.
"""

from __future__ import annotations

from typing import Any

# --- Position Embeddings ---
from .modules.position_embedding import AbsolutePositionEmbedding
from .modules.position_embedding import RoPE
from .modules.position_embedding import SinusoidalPositionalEmbedding

# --- Attention mechanisms ---
from .modules.Attention import Cross_MultiHead_Attention
from .modules.Attention import Group_query_Attention
from .modules.Attention import Group_query_Attention_With_RoPE
from .modules.Attention import Local_Attention
from .modules.Attention import Multi_Head_Attention
from .modules.Attention import Multi_Head_Attention_With_RoPE
from .modules.Attention import Multi_query_Attention
from .modules.Attention import Multi_query_Attention_With_RoPE
from .modules.Attention import Self_Attention
from .modules.Attention import kv_cache_group_query
from .modules.Attention import kv_cache_multihead

# --- Normalization layers ---
from .modules.Normalization import LayerNormalization
from .modules.Normalization import RMSNormalization

# --- Feed Forward layers ---
from .modules.Feed_forward import FF_GELU
from .modules.Feed_forward import FF_GeGLU
from .modules.Feed_forward import FF_LeakyReLU
from .modules.Feed_forward import FF_ReLU
from .modules.Feed_forward import FF_Sigmoid
from .modules.Feed_forward import FF_SiLU
from .modules.Feed_forward import FF_SwiGLU

# --- Models ---
from .models.Google import gemma_1_2b
from .models.Google import gemma_1_7b
from .models.Meta import llama_1
from .models.Meta import llama_2
from .models.OpenAI import GPT_1
from .models.OpenAI import GPT_2
from .models.Transformer import transformer

# --- Vision models ---
from .vision.vit import ViT
from .vision.segformer import SegFormerB0

# --- Generate ---
from .generate import text_generate

# --- Optional public API symbols ---
TokenizerImportError: Exception | None = None
TrainerImportError: Exception | None = None

try:
    from .modules.tokenizer import Embedding_using_tiktoken
except Exception as exc:  # optional dependency: tiktoken
    TokenizerImportError = exc
    Embedding_using_tiktoken = None  # type: ignore[assignment]

try:
    from .trainer import Trainer
except Exception as exc:  # optional dependency: transformers
    TrainerImportError = exc
    Trainer = None  # type: ignore[assignment]

__all__ = [
    "AbsolutePositionEmbedding",
    "Cross_MultiHead_Attention",
    "FF_GELU",
    "FF_GeGLU",
    "FF_LeakyReLU",
    "FF_ReLU",
    "FF_SiLU",
    "FF_Sigmoid",
    "FF_SwiGLU",
    "GPT_1",
    "GPT_2",
    "Group_query_Attention",
    "Group_query_Attention_With_RoPE",
    "LayerNormalization",
    "Local_Attention",
    "Multi_Head_Attention",
    "Multi_Head_Attention_With_RoPE",
    "Multi_query_Attention",
    "Multi_query_Attention_With_RoPE",
    "RMSNormalization",
    "RoPE",
    "Self_Attention",
    "SinusoidalPositionalEmbedding",
    "ViT",
    "gemma_1_2b",
    "gemma_1_7b",
    "kv_cache_group_query",
    "kv_cache_multihead",
    "llama_1",
    "llama_2",
    "text_generate",
    "transformer",
]

if Embedding_using_tiktoken is not None:
    __all__.append("Embedding_using_tiktoken")
if Trainer is not None:
    __all__.append("Trainer")
