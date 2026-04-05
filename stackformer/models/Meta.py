"""Meta-family decoder-only model implementations.

This module provides LLaMA-style causal language models with research-oriented,
yet easy-to-read documentation and usage examples.
"""

import torch
from torch import nn
import torch.nn.functional as F

from stackformer.modules.Attention import Multi_Head_Attention_With_RoPE,kv_cache_group_query
from stackformer.modules.Feed_forward import FF_SwiGLU
from stackformer.modules.Normalization import RMSNormalization
from stackformer.generate import text_generate

# --- llama 1 ---
class llama_1_Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.attention = Multi_Head_Attention_With_RoPE(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        self.norm1 = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        self.FF_SwiGLU = FF_SwiGLU(embed_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm2 = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.FF_SwiGLU(self.norm2(x))
        return x

# --- Decoder ---
class llama_1_Decoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.layers = nn.ModuleList([
            llama_1_Block(embed_dim, num_heads, dropout, hidden_dim, eps, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class llama_1(nn.Module):
    """LLaMA 1-style decoder-only causal language model.

    Simple explanation:
        This model reads token sequences and predicts the next token using only
        left context (causal decoding). It is designed for autoregressive text
        generation.

    Architecture details:
        - Attention: Multi-Head Attention (MHA) with RoPE.
        - Masking: Causal mask.
        - Position encoding: RoPE.
        - Feed-forward: SwiGLU.
        - Normalization: Pre-norm RMSNorm in blocks + final RMSNorm.

    Research context:
        - Family: LLaMA 1 generation of decoder transformer models.
        - Typical use: efficient high-quality language modeling and generation.
        - Paper/report: https://arxiv.org/abs/2302.13971

    Example:
        >>> import torch
        >>> from stackformer.models.Meta import llama_1
        >>> model = llama_1(
        ...     vocab_size=32000,
        ...     num_layers=4,
        ...     embed_dim=512,
        ...     num_heads=8,
        ...     seq_len=128,
        ...     dropout=0.1,
        ...     hidden_dim=2048,
        ... )
        >>> input_ids = torch.randint(0, 32000, (2, 64))
        >>> logits = model(input_ids)
        >>> generated = model.generate(input_ids, max_new_tokens=32)
    """
    def __init__(self, vocab_size, num_layers, embed_dim, num_heads, seq_len,
            dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.seq_len = seq_len
        
        # --- Token embedding ---
        self.embedding = nn.Embedding(vocab_size, embed_dim, dtype=self.dtype, device=self.device)
        
        # --- Decoder ---
        self.decoder = llama_1_Decoder(num_layers=num_layers,embed_dim=embed_dim,num_heads=num_heads,dropout=dropout,
                                    hidden_dim=hidden_dim,eps=eps,device=self.device,dtype=self.dtype)
        
        # --- Final norm        
        self.final_norm = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        
        # --- Output Projection ---
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, dtype=self.dtype, device=self.device)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        emb = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = self.decoder(emb)
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x
        
    @torch.no_grad()
    def generate(self, prompt_ids, max_context_len=128, max_new_tokens=50, temperature=1.0, top_k=None, top_p=1.0, eos_token_id=None):
        return text_generate(self, prompt_ids, max_context_len, max_new_tokens, temperature, top_k, top_p, eos_token_id)

# --- llama 2 ---
class llama_2_Block(nn.Module):
    def __init__(self, embed_dim, num_query_heads, num_kv_heads, batch_size, kv_seq_len, hidden_dim,
                eps=1e-5, dropout=0.1, dtype=torch.float32, device='cpu'):
        super().__init__()
        self.attn_norm = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        self.attn = kv_cache_group_query(embed_dim=embed_dim, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads,
                                        batch_size=batch_size, kv_seq_len=kv_seq_len, dtype=dtype, dropout=dropout, device=device)
        self.ff_norm = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        self.ff = FF_SwiGLU(embed_dim=embed_dim, hidden_dim=hidden_dim, device=device, dtype=dtype)

    def forward(self, x, start_pos):
        x = x + self.attn(self.attn_norm(x), start_pos, rope=True)
        x = x + self.ff(self.ff_norm(x))
        return x

class llama_2_Decoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_query_heads, num_kv_heads, batch_size, kv_seq_len,
                hidden_dim, eps=1e-5, dropout=0.1, dtype=torch.float32, device='cpu'):
        super().__init__()
        self.layers = nn.ModuleList([
            llama_2_Block(embed_dim=embed_dim, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads,
                batch_size=batch_size, kv_seq_len=kv_seq_len, hidden_dim=hidden_dim,
                eps=eps, dropout=dropout, dtype=dtype, device=device)
            for _ in range(num_layers)
        ])

    def forward(self, x, start_pos):
        for layer in self.layers:
            x = layer(x, start_pos)
        return x

class llama_2(nn.Module):
    """LLaMA 2-style decoder-only causal language model with GQA + KV cache.

    Simple explanation:
        This model is an autoregressive decoder that supports grouped-query
        attention and key/value caching. That makes generation faster in long
        decoding loops because cached states are reused.

    Architecture details:
        - Attention: Grouped-Query Attention (GQA) with KV cache. [13B and 70B model]
        - Masking: Causal mask.
        - Position encoding: RoPE (enabled in attention call).
        - Feed-forward: SwiGLU.
        - Normalization: Pre-norm RMSNorm in blocks + final RMSNorm.

    Research context:
        - Family: LLaMA 2 generation decoder transformers.
        - Why important: GQA improves inference efficiency while preserving
          quality compared with full MHA at similar scale.
        - Paper/report: https://arxiv.org/abs/2307.09288

    Example:
        >>> import torch
        >>> from stackformer.models.Meta import llama_2
        >>> model = llama_2(
        ...     num_layers=4,
        ...     embed_dim=512,
        ...     num_query_heads=8,
        ...     num_kv_heads=2,
        ...     batch_size=1,
        ...     kv_seq_len=128,
        ...     vocab_size=32000,
        ...     hidden_dim=2048,
        ... )
        >>> input_ids = torch.randint(0, 32000, (1, 16))
        >>> logits = model(input_ids, start_pos=0)
        >>> generated = model.generate(input_ids, max_new_tokens=24)
    """
    def __init__(self, num_layers, embed_dim, num_query_heads, num_kv_heads, batch_size, kv_seq_len, vocab_size,
                hidden_dim, eps=1e-5, dropout=0.1, dtype=torch.float32, device='cpu'):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.seq_len = kv_seq_len  # For generation slicing

        self.embedding = nn.Embedding(vocab_size, embed_dim, dtype=dtype, device=device)

        self.llama_2_Decoder = llama_2_Decoder(num_layers=num_layers, embed_dim=embed_dim, num_query_heads=num_query_heads,
                            num_kv_heads=num_kv_heads, batch_size=batch_size, kv_seq_len=kv_seq_len,
                            hidden_dim=hidden_dim, eps=eps, dropout=dropout, dtype=dtype, device=device)

        self.final_norm = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, dtype=dtype, device=device)

    def forward(self, input_ids, start_pos=0):
        x = self.embedding(input_ids)
        x = self.llama_2_Decoder(x, start_pos)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_context_len=128, max_new_tokens=50, temperature=1.0, top_k=None, top_p=1.0, eos_token_id=None):
        return text_generate(self, prompt_ids, max_context_len, max_new_tokens, temperature, top_k, top_p, eos_token_id)
