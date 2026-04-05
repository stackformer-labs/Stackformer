"""Google-family decoder-only model implementations.

This module includes two Gemma-style causal language models:
- ``gemma_1_2b`` (Multi-Query Attention variant)
- ``gemma_1_7b`` (Multi-Head Attention variant)

Each model section contains an industrial/research-oriented class docstring with:
- architecture details,
- practical notes,
- simple usage example,
- official paper/report: https://arxiv.org/pdf/2403.08295.
"""

import torch
from torch import nn
import torch.nn.functional as F

from stackformer.modules.Attention import Multi_query_Attention_With_RoPE,Multi_Head_Attention_With_RoPE
from stackformer.modules.Feed_forward import FF_GeGLU
from stackformer.modules.Normalization import RMSNormalization
from stackformer.generate import text_generate

# --- Gemma 1 2B ---
class gemma_1_2b_block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.attention = Multi_query_Attention_With_RoPE(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        self.norm1 = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        self.FF_SwiGLU = FF_GeGLU(embed_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm2 = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.FF_SwiGLU(self.norm2(x))
        return x

# --- Decoder ---
class gemma_1_2b_Decoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.layers = nn.ModuleList([
            gemma_1_2b_block(embed_dim, num_heads, dropout, hidden_dim, eps, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class gemma_1_2b(nn.Module):
    """Gemma 1 2B-style decoder-only causal language model.

    Simple explanation:
        This class builds a GPT-like text generation model that predicts the next
        token from left to right (causal decoding). It uses a stack of decoder
        blocks with pre-normalization, then projects hidden states to vocabulary
        logits.

    Architecture details:
        - Attention: Multi-Query Attention (MQA) with RoPE.
        - Masking: Causal mask (future tokens are hidden).
        - Position encoding: RoPE (Rotary Positional Embedding).
        - Feed-forward: GeGLU.
        - Normalization: Pre-norm RMSNorm in blocks + final RMSNorm.

    Research context:
        - Family: Gemma-style decoder language models.
        - Why used: efficient autoregressive generation with strong scaling
          behavior in modern transformer stacks.
        - Paper/report: https://arxiv.org/pdf/2403.08295

    Example:
        >>> import torch
        >>> from stackformer.models.Google import gemma_1_2b
        >>> model = gemma_1_2b(
        ...     vocab_size=32000,
        ...     num_layers=4,
        ...     embed_dim=512,
        ...     num_heads=8,
        ...     seq_len=128,
        ...     dropout=0.1,
        ...     hidden_dim=2048,
        ... )
        >>> input_ids = torch.randint(0, 32000, (2, 64))
        >>> logits = model(input_ids)  # (batch=2, seq=64, vocab=32000)
        >>> generated = model.generate(input_ids, max_new_tokens=16)
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
        self.decoder = gemma_1_2b_Decoder(num_layers=num_layers,embed_dim=embed_dim,num_heads=num_heads,dropout=dropout,
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

# --- Gemma 1 7B ---
class gemma_1_7b_block(nn.Module):
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
        self.FF_SwiGLU = FF_GeGLU(embed_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm2 = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.FF_SwiGLU(self.norm2(x))
        return x
    
# --- Decoder ---
class gemma_1_7b_Decoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.layers = nn.ModuleList([
            gemma_1_7b_block(embed_dim, num_heads, dropout, hidden_dim, eps, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class gemma_1_7b(nn.Module):
    """Gemma 1 7B-style decoder-only causal language model.

    Simple explanation:
        This is a larger Gemma-style text model that generates text token by
        token. It is very similar to ``gemma_1_2b`` but uses full multi-head
        self-attention in each decoder block.

    Architecture details:
        - Attention: Multi-Head Attention (MHA) with RoPE.
        - Masking: Causal mask.
        - Position encoding: RoPE.
        - Feed-forward: GeGLU.
        - Normalization: Pre-norm RMSNorm in blocks + final RMSNorm.

    Research context:
        - Family: Gemma-style decoder language models.
        - Tradeoff: higher modeling capacity than smaller variants at increased
          memory/compute cost.
        - Paper/report: https://arxiv.org/pdf/2403.08295

    Example:
        >>> import torch
        >>> from stackformer.models.Google import gemma_1_7b
        >>> model = gemma_1_7b(
        ...     vocab_size=32000,
        ...     num_layers=6,
        ...     embed_dim=768,
        ...     num_heads=12,
        ...     seq_len=128,
        ...     dropout=0.1,
        ...     hidden_dim=3072,
        ... )
        >>> input_ids = torch.randint(0, 32000, (1, 32))
        >>> logits = model(input_ids)
        >>> generated = model.generate(input_ids, max_new_tokens=20)
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
        self.decoder = gemma_1_7b_Decoder(num_layers=num_layers,embed_dim=embed_dim,num_heads=num_heads,dropout=dropout,
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
