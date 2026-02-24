"""Attention implementations used by Stackformer.

This module provides a research-to-production set of attention operators:
standard self-attention, RoPE variants, cross-attention, MQA/GQA, local-window
attention, and KV-cache inference attention.

Core equation used by almost all classes:

    Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k) + M) V

where ``M`` is usually a causal mask (``-inf`` on disallowed positions).

Notation:
- ``B``: batch size
- ``T``: query sequence length (current tokens)
- ``S``: key/value sequence length (context or cache length)
- ``C``: embedding dimension
- ``H``: number of query heads
- ``D``: head dimension, usually ``C // H``

Implementation notes:
- Inputs are moved to ``device``/``dtype`` configured in each module.
- Causal masks are cached by sequence shape to reduce allocation overhead.
- RoPE modules require even head dimension (enforced with assertions).

Quick start:
    >>> import torch
    >>> from stackformer.modules.Attention import Multi_Head_Attention
    >>> x = torch.randn(2, 32, 256)
    >>> attn = Multi_Head_Attention(embed_dim=256, num_heads=8)
    >>> y = attn(x, mask=True)
    >>> y.shape
    torch.Size([2, 32, 256])
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from stackformer.modules.Masking import make_mask


class Self_Attention(nn.Module):
    """Single-head causal/self attention.

    Mathematical form:
        - Q = X W_q, K = X W_k, V = X W_v
        - A = softmax((Q K^T) / sqrt(C) + M)
        - Y = A V W_o

    Constructor args:
        embed_dim (int, required): Input/hidden size ``C``.
        dropout (float, optional, default=0.1): Dropout probability on attention
            probabilities after softmax.
        qkv_bias (bool, optional, default=False): Enables bias terms in Q/K/V
            projection layers.
        device (str or torch.device, optional, default='cpu'): Parameter and
            compute device.
        dtype (torch.dtype, optional, default=torch.float32): Parameter and
            compute dtype.

    Forward args:
        x (torch.Tensor, required): Shape ``(B, T, C)``.
        mask (bool, optional, default=True): If True, applies autoregressive
            causal masking (token i cannot attend to tokens > i).

    Returns:
        torch.Tensor: Shape ``(B, T, C)``.

    Example:
        >>> layer = Self_Attention(embed_dim=64, dropout=0.0)
        >>> x = torch.randn(4, 32, 64)
        >>> y = layer(x, mask=True)
    """
    def __init__(self, embed_dim, dropout=0.1, mask_type=['causal'], qkv_bias=False,device='cpu', dtype=torch.float32, **mask_kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.dtype = dtype
        self.mask_type = mask_type
        self.mask_kwargs = mask_kwargs

        # Scaling factor for dot-product attention
        self.scale = 1.0 / math.sqrt(embed_dim)

        # Linear layer
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        
        # Output projection layer to map attention output back to input dimension
        self.out_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)

        # Dropout applied to the attention weights
        self.dropout_p = dropout

        # Cache for causal masks to avoid recomputation across forward passes
        self._causal_mask_cache = {}

    def _get_or_create_mask(self, seq_len: int):
        # Handle boolean mask coming from forward(mask=True)
        if self.mask_type is True:
            mask_types = ["causal"]
        elif self.mask_type in (False, None):
            return None
        elif isinstance(self.mask_type, str):
            mask_types = [self.mask_type]
        elif isinstance(self.mask_type, (list, tuple)):
            mask_types = list(self.mask_type)
        else:
            raise TypeError("mask_type must be bool, str, or list of str")
        
        # Hashable cache key
        key = (seq_len, tuple(mask_types))

        if key not in self._causal_mask_cache:
            mask = make_mask(
                self.mask_type,
                seq_len,
                device=self.device,
                **self.mask_kwargs
            )
            self._causal_mask_cache[key] = mask

        return self._causal_mask_cache[key]

    def forward(self, x, mask=True):
        B, T, C = x.shape
        x = x.to(device=self.device, dtype=self.dtype)

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Add single head dimension for SDPA
        q = q.unsqueeze(1)  # (B, 1, T, C)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        attn_mask = None
        if mask:
            attn_mask = self._get_or_create_mask(T)

        out = F.scaled_dot_product_attention(
            q,k,v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False)

        # Remove head dimension
        out = out.squeeze(1)  # (B, T, C)

        return self.out_proj(out)

class Multi_Head_Attention(nn.Module):
    """Standard multi-head self-attention (MHA).

    Why/when to use:
    - Baseline attention for encoder and decoder blocks.
    - Multiple heads learn different relation subspaces.

    Constructor args:
        embed_dim (int, required): Model width ``C``.
        num_heads (int, required): Number of query heads ``H``.
            Rule: ``embed_dim % num_heads == 0`` (enforced).
        dropout (float, optional, default=0.1): Dropout on attention probs.
        qkv_bias (bool, optional, default=False): Bias in Q/K/V projections.
        device (str or torch.device, optional, default='cpu').
        dtype (torch.dtype, optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): ``(B, T, C)``.
        mask (bool, optional, default=True): Apply causal mask.

    Returns:
        torch.Tensor: ``(B, T, C)``.

    Complexity:
        Time/memory are O(B * H * T^2 * D), dominated by attention matrix.

    Example:
        >>> layer = Multi_Head_Attention(embed_dim=512, num_heads=8)
        >>> y = layer(torch.randn(2, 128, 512), mask=True)
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, mask_type=['causal'],qkv_bias=False,device='cpu', dtype=torch.float32, **mask_kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Each head gets a slice of the embedding
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.mask_type = mask_type
        self.device = device
        self.dtype = dtype
        self.mask_kwargs = mask_kwargs

        # Linear layer
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        
        # Final output projection after attention
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)

        # Dropout applied to the attention weights
        self.dropout_p = dropout
        
        # Cache for causal masks keyed by sequence length
        self._causal_mask_cache = {}

    def _get_or_create_mask(self, seq_len: int):
        # Handle boolean mask coming from forward(mask=True)
        if self.mask_type is True:
            mask_types = ["causal"]
        elif self.mask_type in (False, None):
            return None
        elif isinstance(self.mask_type, str):
            mask_types = [self.mask_type]
        elif isinstance(self.mask_type, (list, tuple)):
            mask_types = list(self.mask_type)
        else:
            raise TypeError("mask_type must be bool, str, or list of str")
        
        # Hashable cache key
        key = (seq_len, tuple(mask_types))

        if key not in self._causal_mask_cache:
            mask = make_mask(
                self.mask_type,
                seq_len,
                device=self.device,
                **self.mask_kwargs
            )
            self._causal_mask_cache[key] = mask

        return self._causal_mask_cache[key]

    def forward(self, x, mask=True):
        B, T, C = x.shape
        x = x.to(device=self.device, dtype=self.dtype)

        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x) 
        v = self.v_proj(x) 
        

        # Reshape for multi-head attention:
        # (B, T, C) → (B, T, num_heads, head_dim) → (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        causal_mask = None
        # Apply causal mask if needed
        if mask:
            causal_mask = self._get_or_create_mask(T)  # (T, T)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final linear projection
        return self.out_proj(out)  # (B, T, C)

class Multi_Head_Attention_With_RoPE(nn.Module):
    """Multi-head self-attention with Rotary Positional Embedding (RoPE).

    RoPE applies a position-dependent 2D rotation on every pair of query/key
    channels, injecting relative position directly into dot products.

    Constructor args:
        embed_dim (int, required).
        num_heads (int, required).
            Rules:
            - ``embed_dim % num_heads == 0``
            - ``head_dim`` must be even for RoPE pair-rotation.
        dropout (float, optional, default=0.1).
        qkv_bias (bool, optional, default=False).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): ``(B, T, C)``.
        mask (bool, optional, default=True).

    Returns:
        torch.Tensor: ``(B, T, C)``.
    """
    def __init__(self, embed_dim, num_heads, mask_type=['causal'], dropout=0.1,  qkv_bias=False,device='cpu', dtype=torch.float32, **mask_kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_type = mask_type
        self.head_dim = embed_dim // num_heads  # Each head gets a slice of the embedding
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.device = device
        self.dtype = dtype
        self.mask_kwargs = mask_kwargs

        # Linear layer
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        
        # Final output projection after attention
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)

        # Dropout applied to the attention weights
        self.dropout_p = dropout
        
        # Cache for causal masks keyed by sequence length
        self._causal_mask_cache = {}

    def _get_or_create_mask(self, seq_len: int):
        # Handle boolean mask coming from forward(mask=True)
        if self.mask_type is True:
            mask_types = ["causal"]
        elif self.mask_type in (False, None):
            return None
        elif isinstance(self.mask_type, str):
            mask_types = [self.mask_type]
        elif isinstance(self.mask_type, (list, tuple)):
            mask_types = list(self.mask_type)
        else:
            raise TypeError("mask_type must be bool, str, or list of str")
        
        # Hashable cache key
        key = (seq_len, tuple(mask_types))

        if key not in self._causal_mask_cache:
            mask = make_mask(
                self.mask_type,
                seq_len,
                device=self.device,
                **self.mask_kwargs
            )
            self._causal_mask_cache[key] = mask

        return self._causal_mask_cache[key]
    
    def _precompute_theta_position_frequency(self, head_dim: int, seq_len: int, theta: float = 10000.0):
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        dim_half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim_half, device=self.device) / dim_half))
        pos = torch.arange(seq_len, device=self.device)
        freqs = torch.outer(pos, inv_freq)  # (seq_len, dim_half)

        freq_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freq_complex  # (seq_len, dim_half)

    def _apply_rotary_position_embedding(self, x: torch.Tensor, freq_complex: torch.Tensor):
        B, H, T, D = x.shape
        assert D % 2 == 0, "head_dim must be even for RoPE"

        x = x.view(B, H, T, D // 2, 2)
        x_complex = torch.view_as_complex(x)  # (B, H, T, D//2)

        freq = freq_complex[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
        x_rot = x_complex * freq  # rotate via complex mult

        x_out = torch.view_as_real(x_rot).view(B, H, T, D)
        return x_out.to(dtype=self.dtype, device=self.device)
    
    def forward(self, x, mask=True):
        B, T, C = x.shape
        x = x.to(device=self.device, dtype=self.dtype)

        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x) 
        v = self.v_proj(x) 
        
        # Reshape for multi-head attention:
        # (B, T, C) → (B, T, num_heads, head_dim) → (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        freq = self._precompute_theta_position_frequency(self.head_dim, T)
        q = self._apply_rotary_position_embedding(q, freq)
        k = self._apply_rotary_position_embedding(k, freq)
        
        causal_mask = None
        # Apply causal mask if needed
        if mask:
            causal_mask = self._get_or_create_mask(T)  # (T, T)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection
        return self.out_proj(out)  # (B, T, C)

class Cross_MultiHead_Attention(nn.Module):
    """Cross-attention: queries from ``x``, keys/values from ``context``.

    Constructor args:
        embed_dim (int, required).
        num_heads (int, required): ``embed_dim % num_heads == 0``.
        dropout (float, optional, default=0.1).
        qkv_bias (bool, optional, default=False).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): Query tensor ``(B, T, C)``.
        context (torch.Tensor): Key/value tensor ``(B, S, C)``.
        mask (bool, optional, default=True): Applies causal mask only when
            ``T == S`` in this implementation.

    Returns:
        torch.Tensor: ``(B, T, C)``.
    """
    def __init__(self, embed_dim, num_heads, mask_type=['causal'],dropout=0.1, qkv_bias=False,device='cpu', dtype=torch.float32, **mask_kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.mask_type = mask_type
        self.device = device
        self.dtype = dtype
        self.mask_kwargs = mask_kwargs

        # Linear layer
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        
        # Dropout applied to the attention weights
        self.dropout_p = dropout
        
        # Cache for causal masks (optional)
        self._causal_mask_cache = {}

    def _get_or_create_mask(self, seq_len: int):
        # Handle boolean mask coming from forward(mask=True)
        if self.mask_type is True:
            mask_types = ["causal"]
        elif self.mask_type in (False, None):
            return None
        elif isinstance(self.mask_type, str):
            mask_types = [self.mask_type]
        elif isinstance(self.mask_type, (list, tuple)):
            mask_types = list(self.mask_type)
        else:
            raise TypeError("mask_type must be bool, str, or list of str")
        
        # Hashable cache key
        key = (seq_len, tuple(mask_types))

        if key not in self._causal_mask_cache:
            mask = make_mask(
                self.mask_type,
                seq_len,
                device=self.device,
                **self.mask_kwargs
            )
            self._causal_mask_cache[key] = mask

        return self._causal_mask_cache[key]

    def forward(self, x, context, mask=True):
        B, T, C = x.shape
        S = context.size(1)  # Length of context sequence

        # Move inputs to the correct device and dtype
        x = x.to(device=self.device, dtype=self.dtype)
        context = context.to(device=self.device, dtype=self.dtype)

        # Compute Q from x, and K/V from context
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(context)
        v = self.v_proj(context)
        
        # Reshape to multi-head format
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, S, head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, S, head_dim)

        causal_mask = None
        # Apply causal mask if needed
        if mask:
            causal_mask = self._get_or_create_mask(T)  # (T, T)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection
        return self.out_proj(out)
    
class Multi_query_Attention(nn.Module):
    """Multi-Query Attention (MQA): many query heads, shared K/V head.

    Constructor args:
        embed_dim (int, required).
        num_heads (int, required): Number of query heads. Rule:
            ``embed_dim % num_heads == 0``.
        dropout (float, optional, default=0.1).
        qkv_bias (bool, optional, default=False).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): ``(B, T, C)``.
        mask (bool, optional, default=True).

    Returns:
        torch.Tensor: ``(B, T, C)``.
    """
    def __init__(self, embed_dim, num_heads, mask_type=['causal'], dropout=0.1, qkv_bias=False,device='cpu', dtype=torch.float32, **mask_kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.mask_type = mask_type
        self.device = device
        self.dtype = dtype
        self.mask_kwargs = mask_kwargs

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        
        # Output final projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        
        # Dropout applied to the attention weights
        self.dropout_p = dropout
        
        # Cache for causal masks (for autoregressive models)
        self._causal_mask_cache = {}

    def _get_or_create_mask(self, seq_len: int):
        # Handle boolean mask coming from forward(mask=True)
        if self.mask_type is True:
            mask_types = ["causal"]
        elif self.mask_type in (False, None):
            return None
        elif isinstance(self.mask_type, str):
            mask_types = [self.mask_type]
        elif isinstance(self.mask_type, (list, tuple)):
            mask_types = list(self.mask_type)
        else:
            raise TypeError("mask_type must be bool, str, or list of str")
        
        # Hashable cache key
        key = (seq_len, tuple(mask_types))

        if key not in self._causal_mask_cache:
            mask = make_mask(
                self.mask_type,
                seq_len,
                device=self.device,
                **self.mask_kwargs
            )
            self._causal_mask_cache[key] = mask

        return self._causal_mask_cache[key]

    def forward(self, x, mask=True):
        B, T, C = x.shape
        x = x.to(device=self.device, dtype=self.q_proj.weight.dtype)

        # Project input to queries (multi-head) and shared keys/values (single head)
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, C)
        v = self.v_proj(x)  # (B, T, C)
        
        # Reshape queries to multi-head: (B, T, C) -> (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Expand shared k/v across heads: (B, T, head_dim) -> (B, 1, T, head_dim)
        # Then broadcast to (B, num_heads, T, head_dim)
        k = k.unsqueeze(1).expand(B, self.num_heads, T, self.head_dim)
        v = v.unsqueeze(1).expand(B, self.num_heads, T, self.head_dim)
        
        causal_mask = None
        # Apply causal mask if needed
        if mask:
            causal_mask = self._get_or_create_mask(T)  # (T, T)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)  # Final linear projection
    
class Multi_query_Attention_With_RoPE(nn.Module):
    """MQA with RoPE on queries and shared keys.

    Constructor args:
        embed_dim (int, required).
        num_heads (int, required): ``embed_dim % num_heads == 0`` and even
            ``head_dim`` for RoPE.
        dropout (float, optional, default=0.1).
        qkv_bias (bool, optional, default=False).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): ``(B, T, C)``.
        mask (bool, optional, default=True).

    Returns:
        torch.Tensor: ``(B, T, C)``.
    """
    def __init__(self, embed_dim, num_heads, mask_type=['causal'],dropout=0.1, qkv_bias=False,device='cpu', dtype=torch.float32, **mask_kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.mask_type = mask_type
        self.device = device
        self.dtype = dtype
        self.mask_kwargs = mask_kwargs

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        
        # Output final projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        
        # Dropout applied to the attention weights
        self.dropout_p = dropout
        
        # Cache for causal masks (for autoregressive models)
        self._causal_mask_cache = {}

    def _get_or_create_mask(self, seq_len: int):
        # Handle boolean mask coming from forward(mask=True)
        if self.mask_type is True:
            mask_types = ["causal"]
        elif self.mask_type in (False, None):
            return None
        elif isinstance(self.mask_type, str):
            mask_types = [self.mask_type]
        elif isinstance(self.mask_type, (list, tuple)):
            mask_types = list(self.mask_type)
        else:
            raise TypeError("mask_type must be bool, str, or list of str")
        
        # Hashable cache key
        key = (seq_len, tuple(mask_types))

        if key not in self._causal_mask_cache:
            mask = make_mask(
                self.mask_type,
                seq_len,
                device=self.device,
                **self.mask_kwargs
            )
            self._causal_mask_cache[key] = mask

        return self._causal_mask_cache[key]
    
    def _precompute_theta_position_frequency(self, head_dim: int, seq_len: int, theta: float = 10000.0):
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        dim_half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim_half, device=self.device) / dim_half))
        pos = torch.arange(seq_len, device=self.device)
        freqs = torch.outer(pos, inv_freq)  # (seq_len, dim_half)

        freq_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freq_complex  # (seq_len, dim_half)

    def _apply_rotary_position_embedding(self, x: torch.Tensor, freq_complex: torch.Tensor):
        B, H, T, D = x.shape
        assert D % 2 == 0, "head_dim must be even for RoPE"

        x = x.view(B, H, T, D // 2, 2)
        x_complex = torch.view_as_complex(x)  # (B, H, T, D//2)

        freq = freq_complex[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
        x_rot = x_complex * freq  # rotate via complex mult

        x_out = torch.view_as_real(x_rot).view(B, H, T, D)
        return x_out.to(dtype=self.dtype, device=self.device)
    
    def forward(self, x, mask=True):
        B, T, C = x.shape
        x = x.to(device=self.device, dtype=self.q_proj.weight.dtype)

        # Project input to queries (multi-head) and shared keys/values (single head)
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, C)
        v = self.v_proj(x)  # (B, T, C)
        
        # Reshape queries to multi-head: (B, T, C) -> (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Expand shared k/v across heads: (B, T, head_dim) -> (B, 1, T, head_dim)
        # Then broadcast to (B, num_heads, T, head_dim)
        k = k.unsqueeze(1).expand(B, self.num_heads, T, self.head_dim)
        v = v.unsqueeze(1).expand(B, self.num_heads, T, self.head_dim)
        
        freq = self._precompute_theta_position_frequency(self.head_dim, T)
        q = self._apply_rotary_position_embedding(q, freq)
        k = self._apply_rotary_position_embedding(k, freq)
        
        causal_mask = None
        # Apply causal mask if needed
        if mask:
            causal_mask = self._get_or_create_mask(T)  # (T, T)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)  # Final linear projection

class Group_query_Attention(nn.Module):
    """Grouped-Query Attention (GQA): intermediate between MHA and MQA.

    Constructor args:
        embed_dim (int, required).
        num_query_heads (int, required): Rule ``embed_dim % num_query_heads == 0``.
        num_kv_heads (int, required): Rule ``num_query_heads % num_kv_heads == 0``.
        qkv_bias (bool, optional, default=False).
        dropout (float, optional, default=0.1).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): ``(B, T, C)``.
        mask (bool, optional, default=True).

    Returns:
        torch.Tensor: ``(B, T, C)``.
    """
    def __init__(self, embed_dim, num_query_heads, num_kv_heads, qkv_bias=False, mask_type=['causal'],dropout=0.1, device='cpu', dtype=torch.float32, **mask_kwargs):
        super().__init__()
        assert embed_dim % num_query_heads == 0, "embed_dim must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        self.dtype = dtype
        self.device = device
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.mask_type = mask_type
        self.device = device
        self.mask_kwargs = mask_kwargs

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=qkv_bias, device=device, dtype=dtype)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        
        # Dropout applied to the attention weights
        self.dropout_p = dropout
        
        self._causal_mask_cache = {}

    def _get_or_create_mask(self, seq_len: int):
        # Handle boolean mask coming from forward(mask=True)
        if self.mask_type is True:
            mask_types = ["causal"]
        elif self.mask_type in (False, None):
            return None
        elif isinstance(self.mask_type, str):
            mask_types = [self.mask_type]
        elif isinstance(self.mask_type, (list, tuple)):
            mask_types = list(self.mask_type)
        else:
            raise TypeError("mask_type must be bool, str, or list of str")
        
        # Hashable cache key
        key = (seq_len, tuple(mask_types))

        if key not in self._causal_mask_cache:
            mask = make_mask(
                self.mask_type,
                seq_len,
                device=self.device,
                **self.mask_kwargs
            )
            self._causal_mask_cache[key] = mask

        return self._causal_mask_cache[key]

    def forward(self, x, mask=True):
        B, T, C = x.shape
        x = x.to(device=self.device, dtype=self.q_proj.weight.dtype)

        # Project Q, K, V
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x) 
        v = self.v_proj(x) 

        # Reshape projections
        q = q.view(B, T, self.num_query_heads, self.head_dim).transpose(1, 2)  # (B, num_query_heads, T, head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)     # (B, num_kv_heads, T, head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat keys and values for each query head group
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)  # (B, num_query_heads, T, head_dim)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        causal_mask = None
        # Apply causal mask if needed
        if mask:
            causal_mask = self._get_or_create_mask(T)  # (T, T)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)
    
class Group_query_Attention_With_RoPE(nn.Module):
    """GQA with RoPE for relative-position-aware grouped attention.

    Constructor args:
        embed_dim (int, required).
        num_query_heads (int, required): ``embed_dim % num_query_heads == 0``.
        num_kv_heads (int, required): ``num_query_heads % num_kv_heads == 0``.
        qkv_bias (bool, optional, default=False).
        dropout (float, optional, default=0.1).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Rules:
        RoPE requires even head dimension.

    Forward args:
        x (torch.Tensor): ``(B, T, C)``.
        mask (bool, optional, default=True).

    Returns:
        torch.Tensor: ``(B, T, C)``.
    """
    def __init__(self, embed_dim, num_query_heads, num_kv_heads, qkv_bias=False, mask_type=['causal'], dropout=0.1, device='cpu', dtype=torch.float32, **mask_kwargs):
        super().__init__()
        assert embed_dim % num_query_heads == 0, "embed_dim must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        self.dtype = dtype
        self.device = device
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.mask_type = mask_type
        self.device = device
        self.mask_kwargs = mask_kwargs

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=qkv_bias, device=device, dtype=dtype)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        
        # Dropout applied to the attention weights
        self.dropout_p = dropout

        self._causal_mask_cache = {}

    def _get_or_create_mask(self, seq_len: int):
        # Handle boolean mask coming from forward(mask=True)
        if self.mask_type is True:
            mask_types = ["causal"]
        elif self.mask_type in (False, None):
            return None
        elif isinstance(self.mask_type, str):
            mask_types = [self.mask_type]
        elif isinstance(self.mask_type, (list, tuple)):
            mask_types = list(self.mask_type)
        else:
            raise TypeError("mask_type must be bool, str, or list of str")
        
        # Hashable cache key
        key = (seq_len, tuple(mask_types))

        if key not in self._causal_mask_cache:
            mask = make_mask(
                self.mask_type,
                seq_len,
                device=self.device,
                **self.mask_kwargs
            )
            self._causal_mask_cache[key] = mask

        return self._causal_mask_cache[key]
    
    def _precompute_theta_position_frequency(self, head_dim: int, seq_len: int, theta: float = 10000.0):
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        dim_half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim_half, device=self.device) / dim_half))
        pos = torch.arange(seq_len, device=self.device)
        freqs = torch.outer(pos, inv_freq)  # (seq_len, dim_half)

        freq_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freq_complex  # (seq_len, dim_half)

    def _apply_rotary_position_embedding(self, x: torch.Tensor, freq_complex: torch.Tensor):
        B, H, T, D = x.shape
        assert D % 2 == 0, "head_dim must be even for RoPE"

        x = x.view(B, H, T, D // 2, 2)
        x_complex = torch.view_as_complex(x)  # (B, H, T, D//2)

        freq = freq_complex[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
        x_rot = x_complex * freq  # rotate via complex mult

        x_out = torch.view_as_real(x_rot).view(B, H, T, D)
        return x_out.to(dtype=self.dtype, device=self.device)
    
    def forward(self, x, mask=True):
        B, T, C = x.shape
        x = x.to(device=self.device, dtype=self.q_proj.weight.dtype)

        # Project Q, K, V
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x) 
        v = self.v_proj(x) 

        # Reshape projections
        q = q.view(B, T, self.num_query_heads, self.head_dim).transpose(1, 2)  # (B, num_query_heads, T, head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)     # (B, num_kv_heads, T, head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat keys and values for each query head group
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)  # (B, num_query_heads, T, head_dim)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        freq = self._precompute_theta_position_frequency(self.head_dim, T)
        q = self._apply_rotary_position_embedding(q, freq)
        k = self._apply_rotary_position_embedding(k, freq)
        
        causal_mask = None
        # Apply causal mask if needed
        if mask:
            causal_mask = self._get_or_create_mask(T)  # (T, T)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)

class Local_Attention(nn.Module):
    """Sliding-window causal self-attention.

    Constructor args:
        embed_dim (int, required).
        num_heads (int, required): Rule ``embed_dim % num_heads == 0``.
        window_size (int, required): Rule ``window_size >= 1``.
        qkv_bias (bool, optional, default=False).
        dropout (float, optional, default=0.1).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): ``(B, T, C)``.
        mask (bool, optional, default=True): Apply local causal mask.

    Returns:
        torch.Tensor: ``(B, T, C)``.
    """
    def __init__(self, embed_dim, num_heads, window_size, mask_type=['sliding_window'], qkv_bias=False,dropout=0.1, device='cpu', dtype=torch.float32):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert window_size >= 1, "Window size must be >= 1 to avoid full masking"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.device = device
        self.dtype = dtype
        self.mask_type = mask_type

        # QKV projection
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        
        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=self.dtype)
        
        # Dropout applied to the attention weights
        self.dropout_p = dropout

        # Cache causal masks for efficiency
        self._causal_mask_cache = {}

    def _get_or_create_mask(self, seq_len: int, window_size: int):
        key = (seq_len)
        if key not in self._causal_mask_cache:
            mask = make_mask(
                self.mask_type,
                seq_len,
                window_size=window_size,
                device=self.device,
            )
            self._causal_mask_cache[key] = mask

        return self._causal_mask_cache[key]

    def forward(self, x, mask=True):
        B, T, C = x.shape
        x = x.to(device=self.device, dtype=self.q_proj.weight.dtype)

        # Project Q, K, V
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x) 
        v = self.v_proj(x) 

        # Split heads
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        causal_mask = None
        # Apply causal mask if needed
        if mask:
            causal_mask = self._get_or_create_mask(T, window_size=self.window_size)  # (T, T)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)

class kv_cache_multihead(nn.Module):
    """MHA with persistent KV cache for incremental decoding.

    Constructor args:
        embed_dim (int, required).
        num_heads (int, required): ``embed_dim % num_heads == 0``.
        batch_size (int, required): Preallocated cache batch capacity.
        kv_seq_len (int, required): Maximum cache sequence length reference.
        qkv_bias (bool, optional, default=False).
        dropout (float, optional, default=0.1).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): ``(B, T, C)`` token chunk.
        start_pos (int, required): Write offset in cache.
        mask (bool, optional, default=True): Causal masking over cache span.
        rope (bool, optional, default=True): Enable RoPE before cache write.

    Returns:
        torch.Tensor: ``(B, T, C)``.
    """
    def __init__(self, embed_dim, num_heads, batch_size, kv_seq_len, mask_type=['causal'], qkv_bias=False, dropout=0.1, device='cpu', dtype=torch.float32, **mask_kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.device = device
        self.dtype = dtype
        self.mask_type = mask_type
        self.mask_kwargs = mask_kwargs
        
        #  QKV projection
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        
        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        
        # Dropout applied to the attention weights
        self.dropout_p = dropout

        # KV Cache (double kv_seq_len to avoid OOB)
        self.cache_keys = torch.zeros(batch_size, kv_seq_len * 2, num_heads, self.head_dim, device=device, dtype=dtype)
        self.cache_values = torch.zeros(batch_size, kv_seq_len * 2, num_heads, self.head_dim, device=device, dtype=dtype)

        self._causal_mask_cache = {}

    def _get_or_create_kv_mask(self, T: int, S: int, start_pos: int):
        key = (T, S, start_pos)

        if key not in self._causal_mask_cache:
            i = torch.arange(T, device=self.device).unsqueeze(1)
            j = torch.arange(S, device=self.device).unsqueeze(0)
            visible = j <= (start_pos + i)
            self._causal_mask_cache[key] = ~visible

        return self._causal_mask_cache[key]

    def _precompute_theta_position_frequency(self, head_dim: int, seq_len: int, theta: float = 10000.0):
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        dim_half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim_half, device=self.device) / dim_half))
        pos = torch.arange(seq_len, device=self.device)
        freqs = torch.outer(pos, inv_freq)  # (seq_len, dim_half)

        freq_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freq_complex  # (seq_len, dim_half)

    def _apply_rotary_position_embedding(self, x: torch.Tensor, freq_complex: torch.Tensor):
        B, H, T, D = x.shape
        assert D % 2 == 0, "head_dim must be even for RoPE"

        x = x.view(B, H, T, D // 2, 2)
        x_complex = torch.view_as_complex(x)  # (B, H, T, D//2)

        freq = freq_complex[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
        x_rot = x_complex * freq  # rotate via complex mult

        x_out = torch.view_as_real(x_rot).view(B, H, T, D)
        return x_out.to(dtype=self.dtype, device=self.device)

    def forward(self, x: torch.Tensor, start_pos: int, mask: bool = True, rope: bool = True):
        B, T, C = x.shape
        assert C == self.embed_dim, "Input embed_dim mismatch"

        # Project Q, K, V
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to multi-head
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        
        # Rotary Position Embedding (correct for KV cache)        
        if rope:
            end_pos = start_pos + T
            freq = self._precompute_theta_position_frequency(self.head_dim, end_pos)

            # apply only slice corresponding to current tokens
            q = self._apply_rotary_position_embedding(q, freq[start_pos:end_pos])
            k = self._apply_rotary_position_embedding(k, freq[start_pos:end_pos])
        else:
            end_pos = start_pos + T

        # Write KV cache
        self.cache_keys[:B, start_pos:end_pos] = k.detach()
        self.cache_values[:B, start_pos:end_pos] = v.detach()
        
        # Read full cached KV        
        k_full = self.cache_keys[:B, :end_pos].detach()     # (B, S, H, D)
        v_full = self.cache_values[:B, :end_pos].detach()

        # Transpose to SDPA format        
        q = q.transpose(1, 2)           # (B, H, T, D)
        k_full = k_full.transpose(1, 2)  # (B, H, S, D)
        v_full = v_full.transpose(1, 2)

        
        # Rectangular causal mask (T,S)
        attn_mask = None
        if mask:
            attn_mask = self._get_or_create_kv_mask(T, end_pos, start_pos)

        # Scaled Dot Product Attention (Flash / MemEff)        
        context = F.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            attn_mask=attn_mask,   # (T, S)
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )

        # Merge heads + output projection        
        out = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class kv_cache_group_query(nn.Module):  
    """GQA with KV cache for production-grade decoding throughput.

    Constructor args:
        embed_dim (int, required).
        num_query_heads (int, required): ``embed_dim % num_query_heads == 0``.
        num_kv_heads (int, required): ``num_query_heads % num_kv_heads == 0``.
        kv_seq_len (int, required): Maximum cache length reference.
        batch_size (int, required): Cache batch capacity.
        qkv_bias (bool, optional, default=False).
        dropout (float, optional, default=0.1).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): ``(B, T, C)``.
        start_pos (int, required): Cache write position.
        mask (bool, optional, default=True).
        rope (bool, optional, default=True).

    Returns:
        torch.Tensor: ``(B, T, C)``.
    """
    def __init__(self, embed_dim, num_query_heads, num_kv_heads, kv_seq_len, batch_size, mask_type=['causal'], qkv_bias=False, dropout=0.1, device='cpu', dtype=torch.float32, **mask_kwargs):
        super().__init__()
        assert embed_dim % num_query_heads == 0, "embed_dim must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.kv_seq_len = kv_seq_len
        self.device = device
        self.dtype = dtype
        self.mask_type = mask_type
        self.mask_kwargs = mask_kwargs

        # Linear projections: Q from full dim, KV from reduced dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=qkv_bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=qkv_bias, device=device, dtype=dtype)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        
        # Dropout applied to the attention weights
        self.dropout_p = dropout
        
        self._causal_mask_cache = {}

        # KV Cache (double kv_seq_len to avoid OOB)
        self.cache_keys = torch.zeros(batch_size, kv_seq_len * 2, num_kv_heads, self.head_dim, device=device, dtype=dtype)
        self.cache_values = torch.zeros(batch_size, kv_seq_len * 2, num_kv_heads, self.head_dim, device=device, dtype=dtype)

    def _get_or_create_kv_mask(self, T: int, S: int, start_pos: int):
        key = (T, S, start_pos)

        if key not in self._causal_mask_cache:
            i = torch.arange(T, device=self.device).unsqueeze(1)
            j = torch.arange(S, device=self.device).unsqueeze(0)
            visible = j <= (start_pos + i)
            self._causal_mask_cache[key] = ~visible

        return self._causal_mask_cache[key]

    def _precompute_theta_position_frequency(self, head_dim: int, seq_len: int, theta: float = 10000.0):
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        dim_half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim_half, device=self.device) / dim_half))
        pos = torch.arange(seq_len, device=self.device)
        freqs = torch.outer(pos, inv_freq)  # (seq_len, dim_half)
        freq_complex = torch.polar(torch.ones_like(freqs), freqs)  # magnitude=1, angle=freqs
        return freq_complex  # (seq_len, dim_half)

    def _apply_rotary_position_embedding(self, x: torch.Tensor, freq_complex: torch.Tensor):
        B, H, T, D = x.shape
        assert D % 2 == 0, "head_dim must be even for RoPE"

        x = x.view(B, H, T, D // 2, 2)  # Split last dim into real+imag
        x_complex = torch.view_as_complex(x)  # (B, H, T, D//2)
        freq = freq_complex[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
        x_rot = x_complex * freq  # Complex multiplication = rotation
        x_out = torch.view_as_real(x_rot).view(B, H, T, D)  # Back to real tensor
        return x_out.to(dtype=self.dtype, device=self.device)

    def forward(self, x, start_pos, mask=True, rope=True):
        B, T, C = x.shape
        x = x.to(device=self.device, dtype=self.q_proj.weight.dtype)

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Reshape
        # Q: (B, T, QH, D)
        # K/V: (B, T, KVH, D)
        q = q.view(B, T, self.num_query_heads, self.head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim)
        end_pos = start_pos + T

        # RoPE (correct for KV cache)
        if rope:
            freq = self._precompute_theta_position_frequency(self.head_dim, end_pos)
            q = self._apply_rotary_position_embedding(
                q, freq[start_pos:end_pos]
            )
            k = self._apply_rotary_position_embedding(
                k, freq[start_pos:end_pos]
            )

        # Write KV cache
        self.cache_keys[:B, start_pos:end_pos] = k.detach()
        self.cache_values[:B, start_pos:end_pos] = v.detach()

        # Read full KV
        k_full = self.cache_keys[:B, :end_pos].detach()   # (B, S, KVH, D)
        v_full = self.cache_values[:B, :end_pos].detach()

        # Transpose to attention format
        q = q.transpose(1, 2)            # (B, QH, T, D)
        k_full = k_full.transpose(1, 2)  # (B, KVH, S, D)
        v_full = v_full.transpose(1, 2)

        # Expand KV heads to match Q heads (GQA)
        k_full = k_full.repeat_interleave(self.num_queries_per_kv, dim=1)
        v_full = v_full.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Rectangular causal mask (T,S)
        attn_mask = None
        if mask:
            attn_mask = self._get_or_create_kv_mask(T, end_pos, start_pos)

        # SDPA
        context = F.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )

        # Merge heads
        out = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)