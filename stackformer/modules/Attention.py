"""
Attention Mechanisms for StackFormer Library

This module provides a comprehensive suite of attention mechanisms used in modern 
transformer architectures. It includes various self-attention and cross-attention 
modules such as:

- Single-head and Multi-head Self-Attention
- Multi-head Attention with Rotary Positional Embeddings (RoPE)
- Multi-Query and Grouped Query Attention (MQA, GQA)
- Linear and Local Attention for efficient long-sequence modeling
- Cross-Attention for encoder-decoder architectures
- Attention with Key/Value (KV) caching for autoregressive decoding
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Self_Attention(nn.Module):
    """
    Implements a single-head self-attention mechanism with fused QKV projection,
    optional causal masking, and dropout for regularization.

    Args:
        embed_dim (int): The input and output dimensionality of the attention mechanism.
        dropout (float): Dropout probability applied after the softmax attention weights.
        device (str): Device to use for tensors ('cpu' or 'cuda').
        dtype (torch.dtype): Data type for the projection layers (default: torch.float32).
    """
    def __init__(self, embed_dim, dropout=0.1, device='cpu', dtype=torch.float32):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.dtype = dtype

        # Scaling factor for dot-product attention
        self.scale = 1.0 / math.sqrt(embed_dim)

        # Linear layer that computes Q, K, V in one go: [B, T, 3*embed_dim]
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False, device=device, dtype=dtype)
        
        # Output projection layer to map attention output back to input dimension
        self.out_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)

        # Dropout applied to the attention weights
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

        # Cache for causal masks to avoid recomputation across forward passes
        self._causal_mask_cache = {}

    def _get_or_create_causal_mask(self, seq_len):
        """
        Retrieves or creates a causal attention mask of shape (seq_len, seq_len),
        where positions in the upper triangle (i < j) are masked with True.

        Args:
            seq_len (int): Length of the sequence.

        Returns:
            torch.Tensor: A boolean causal mask tensor.
        """
        if seq_len not in self._causal_mask_cache:
            # Upper triangular mask with True above the diagonal
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device),
                diagonal=1
            )
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]

    def forward(self, x, mask=True):
        """
        Performs self-attention over the input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where
                            B = batch size, T = sequence length, C = embed_dim.
            mask (bool): If True, applies a causal mask to prevent attention to future tokens.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        B, T, C = x.shape

        x = x.to(device=self.device, dtype=self.qkv_proj.weight.dtype)
        # Compute queries, keys, and values
        qkv = self.qkv_proj(x)  # Shape: (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)  # Each of shape: (B, T, C)

        # Compute raw attention scores (dot product of Q and K^T)
        att = (q @ k.transpose(1, 2)) * self.scale  # Shape: (B, T, T)

        # Apply causal mask to prevent attending to future positions
        if mask:
            causal_mask = self._get_or_create_causal_mask(T)  # Shape: (T, T)
            att.masked_fill_(causal_mask[None, :, :], float('-inf'))

        # Normalize attention scores using softmax
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Compute attention output as weighted sum of values
        out = att @ v  # Shape: (B, T, C)

        # Apply final output projection
        return self.out_proj(out)  # Shape: (B, T, C)

class Multi_Head_Attention(nn.Module):
    """
    Implements multi-head self-attention with fused QKV projection and optional causal masking.
    
    Each input token attends to all other tokens (or only past tokens if causal masking is used),
    using multiple attention heads in parallel.

    Args:
        embed_dim (int): Total embedding dimension of the model (divided among heads).
        num_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability applied to the attention weights.
        device (str): Device to store tensors and layers ('cpu' or 'cuda').
        dtype (torch.dtype): Data type used for weights and computations (default: torch.float32).
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, device='cpu', dtype=torch.float32):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Each head gets a slice of the embedding
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.device = device

        # Fused linear layer to compute Q, K, V simultaneously: [B, T, 3*embed_dim]
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False, device=device, dtype=dtype)
        
        # Final output projection after attention
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        
        self.dropout = nn.Dropout(dropout)

        # Cache for causal masks keyed by sequence length
        self._causal_mask_cache = {}

    def _get_or_create_causal_mask(self, seq_len):
        """
        Generates or retrieves a cached causal attention mask of shape (seq_len, seq_len).
        Ensures that tokens can only attend to themselves and previous tokens.

        Args:
            seq_len (int): Length of the sequence.

        Returns:
            torch.Tensor: Boolean mask where True means "mask out".
        """
        if seq_len not in self._causal_mask_cache:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device),
                diagonal=1
            )
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]

    def forward(self, x, mask=True):
        """
        Computes the multi-head self-attention output for input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where
                            B = batch size, T = sequence length, C = embed_dim.
            mask (bool): If True, applies a causal mask to prevent attention to future tokens.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        B, T, C = x.shape

        # Ensure the input is on the correct device and dtype
        x = x.to(device=self.device, dtype=self.qkv_proj.weight.dtype)

        # Project to queries, keys, values: shape (B, T, 3*C)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (B, T, C)

        # Reshape for multi-head attention:
        # (B, T, C) → (B, T, num_heads, head_dim) → (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention scores
        att = (q @ k.transpose(-2, -1)) * self.scale  # Shape: (B, num_heads, T, T)

        # Apply causal mask if needed
        if mask:
            causal_mask = self._get_or_create_causal_mask(T)  # (T, T)
            att.masked_fill_(causal_mask[None, None, :, :], float('-inf'))

        # Normalize scores and apply dropout
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Compute attention output
        out = att @ v  # (B, num_heads, T, head_dim)

        # Concatenate all heads: (B, T, num_heads * head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection
        return self.out_proj(out)  # (B, T, C)

class Multi_Head_Attention_with_RoPE(nn.Module):
    """
    Implements multi-head self-attention with Rotary Positional Embeddings (RoPE)
    for encoding relative positional information. Supports causal masking and dropout.

    Args:
        embed_dim (int): Total embedding dimension of the model (divided among heads).
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability applied to attention scores.
        device (str): Device for computation ('cpu' or 'cuda').
        dtype (torch.dtype): Data type for model weights and activations.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, device='cpu', dtype=torch.float32):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.device = device
        self.dtype = dtype

        # Linear layer to compute Q, K, V in one shot
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False, device=device, dtype=dtype)

        # Output projection layer
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)

        self.dropout = nn.Dropout(dropout)

        # Cache for causal attention masks
        self._causal_mask_cache = {}

    def _get_or_create_causal_mask(self, seq_len):
        """
        Returns a cached or newly created causal mask for attention.

        Args:
            seq_len (int): Sequence length.

        Returns:
            torch.Tensor: A boolean tensor of shape (seq_len, seq_len).
        """
        if seq_len not in self._causal_mask_cache:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device),
                diagonal=1
            )
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]

    def _precompute_theta_position_frequency(self, head_dim, seq_len, theta=10000.0):
        """
        Precomputes rotary position encodings as complex exponentials.

        Args:
            head_dim (int): Dimension of each attention head.
            seq_len (int): Sequence length.
            theta (float): Base for exponential frequency (default: 10000).

        Returns:
            torch.Tensor: Complex tensor of shape (seq_len, head_dim // 2)
        """
        assert head_dim % 2 == 0, "head_dim must be even for complex RoPE"

        # Inverse frequency terms
        theta_numerator = torch.arange(0, head_dim, 2, device=self.device)
        inv_freq = 1.0 / (theta ** (theta_numerator / head_dim))

        # Outer product of positions and frequencies
        positions = torch.arange(seq_len, device=self.device)
        freqs = torch.outer(positions, inv_freq)

        # Convert to complex exponential form: exp(i * freq)
        freq_complex = torch.polar(torch.ones_like(freqs), freqs)  # Shape: (seq_len, head_dim // 2)
        return freq_complex

    def _apply_rotary_position_embedding(self, x, freq_complex):
        """
        Applies RoPE to input tensor using precomputed frequencies.

        Args:
            x (torch.Tensor): Input tensor of shape (B, num_heads, T, head_dim)
            freq_complex (torch.Tensor): Complex frequencies of shape (T, head_dim//2)

        Returns:
            torch.Tensor: Tensor with RoPE applied, same shape as input
        """
        B, num_heads, T, dim = x.shape
        assert dim % 2 == 0, "head_dim must be even to apply RoPE"

        # Convert to complex by pairing dimensions
        x = x.view(B, num_heads, T, dim // 2, 2)
        x_complex = torch.view_as_complex(x)  # Shape: (B, num_heads, T, dim // 2)

        # Expand freq to match x_complex shape
        freq = freq_complex[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, dim // 2)

        # Elementwise complex multiplication (rotation)
        x_rotated = x_complex * freq  # (B, num_heads, T, dim // 2)

        # Convert back to real
        x_out = torch.view_as_real(x_rotated).view(B, num_heads, T, dim)
        return x_out.to(dtype=self.dtype, device=self.device)

    def forward(self, x, mask=True):
        """
        Forward pass for multi-head attention with RoPE.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).
            mask (bool): Whether to apply causal attention mask.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        B, T, C = x.shape
        x = x.to(dtype=self.dtype, device=self.device)

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (B, T, C)

        # Reshape for multi-head attention: (B, T, C) → (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Precompute rotary frequencies and apply RoPE
        freqs = self._precompute_theta_position_frequency(self.head_dim, T)
        q = self._apply_rotary_position_embedding(q, freqs)
        k = self._apply_rotary_position_embedding(k, freqs)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, T, T)

        # Apply causal mask if needed
        if mask:
            causal_mask = self._get_or_create_causal_mask(T)  # (T, T)
            att.masked_fill_(causal_mask[None, None, :, :], float('-inf'))

        # Normalize and apply dropout
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Weighted sum of V
        out = att @ v  # (B, num_heads, T, head_dim)

        # Merge heads and apply final projection
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class Cross_MultiHead_Attention(nn.Module):
    """
    Implements multi-head cross-attention where the query comes from the input `x`,
    and the key/value pairs come from a separate `context` input.
    
    Useful for encoder-decoder attention in Transformer models.

    Args:
        embed_dim (int): Total embedding dimension (shared across Q, K, V).
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability on attention weights.
        device (str): Device to use for tensors and parameters.
        dtype (torch.dtype): Data type for model parameters and activations.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, device='cpu', dtype=torch.float32):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.device = device
        self.dtype = dtype

        # Linear projections for query and key/value (separate sources)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False, device=device, dtype=dtype)

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        # Cache for causal masks (optional)
        self._causal_mask_cache = {}

    def _get_or_create_causal_mask(self, seq_len):
        """
        Retrieves or creates a causal attention mask (upper triangular) to
        prevent attending to future tokens.

        Args:
            seq_len (int): Sequence length.

        Returns:
            torch.Tensor: A boolean mask of shape (seq_len, seq_len).
        """
        if seq_len not in self._causal_mask_cache:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device),
                diagonal=1
            )
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]

    def forward(self, x, context, mask=True):
        """
        Performs cross-attention where queries come from `x` and keys/values from `context`.

        Args:
            x (torch.Tensor): Query input of shape (B, T, C).
            context (torch.Tensor): Key/value source input of shape (B, S, C).
            mask (bool): If True, applies causal masking to attention weights.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        B, T, C = x.shape
        S = context.size(1)  # Length of context sequence

        # Move inputs to the correct device and dtype
        x = x.to(device=self.device, dtype=self.dtype)
        context = context.to(device=self.device, dtype=self.dtype)

        # Compute Q from x, and K/V from context
        q = self.q_proj(x)  # (B, T, C)
        kv = self.kv_proj(context)  # (B, S, 2*C)
        k, v = kv.chunk(2, dim=-1)  # Each: (B, S, C)

        # Reshape to multi-head format
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, S, head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, S, head_dim)

        # Compute scaled attention scores: Q × K^T
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, T, S)

        # Optionally apply causal mask (usually not used in cross-attention)
        if mask and T == S:
            causal_mask = self._get_or_create_causal_mask(T)
            att.masked_fill_(causal_mask[None, None, :, :], float('-inf'))

        # Softmax normalization and dropout
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Attention output: A × V
        out = att @ v  # (B, num_heads, T, head_dim)

        # Combine heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection
        return self.out_proj(out)
    
class Multi_query_Attention(nn.Module):
    """
    Implements Multi-Query Attention (MQA), where a single set of keys and values
    are shared across all attention heads, but each head has its own set of queries.

    This improves memory and compute efficiency compared to full MHA, especially
    in decoder-only architectures like GPT.

    Args:
        embed_dim (int): Total input/output embedding dimension.
        num_heads (int): Number of attention heads (applied to queries only).
        dropout (float): Dropout probability for attention weights.
        device (str): Device on which the module will run ('cpu' or 'cuda').
        dtype (torch.dtype): Data type for model weights and activations.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, device='cpu', dtype=torch.float32):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.device = device

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.kv_proj = nn.Linear(embed_dim, 2 * self.head_dim, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)

        self.dropout = nn.Dropout(dropout)

        # Cache for causal masks (for autoregressive models)
        self._causal_mask_cache = {}

    def _get_or_create_causal_mask(self, seq_len):
        """
        Retrieves or creates a causal attention mask (upper triangular) to
        prevent attending to future tokens.

        Args:
            seq_len (int): Sequence length.

        Returns:
            torch.Tensor: A boolean mask of shape (seq_len, seq_len).
        """
        if seq_len not in self._causal_mask_cache:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device),
                diagonal=1
            )
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]

    def forward(self, x, mask=True):
        """
        Forward pass of the Multi-Query Attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).
            mask (bool): Whether to apply causal masking (default: True).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        B, T, C = x.shape
        x = x.to(device=self.device, dtype=self.q_proj.weight.dtype)

        # Project input to queries (multi-head) and shared keys/values (single head)
        q = self.q_proj(x)  # (B, T, C)
        kv = self.kv_proj(x)  # (B, T, 2 * head_dim)
        k, v = kv.chunk(2, dim=-1)  # Each: (B, T, head_dim)

        # Reshape queries to multi-head: (B, T, C) -> (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Expand shared k/v across heads: (B, T, head_dim) -> (B, 1, T, head_dim)
        # Then broadcast to (B, num_heads, T, head_dim)
        k = k.unsqueeze(1).expand(B, self.num_heads, T, self.head_dim)
        v = v.unsqueeze(1).expand(B, self.num_heads, T, self.head_dim)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, T, T)

        # Apply causal mask if requested
        if mask:
            causal_mask = self._get_or_create_causal_mask(T)  # (T, T)
            att = att.masked_fill(causal_mask[None, None, :, :], float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Attention output
        out = att @ v  # (B, num_heads, T, head_dim)

        # Merge heads back: (B, num_heads, T, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)  # Final linear projection
    
class Group_query_Attention(nn.Module):
    """
    Implements Grouped Query Attention (GQA), where the number of query heads 
    can be higher than the number of key/value heads.
    
    This improves memory and compute efficiency while retaining expressiveness.
    """

    def __init__(self, embed_dim, num_query_heads, num_kv_heads, dropout=0.1, device='cpu', dtype=torch.float32):
        """
        Args:
            embed_dim (int): Total embedding dimension.
            num_query_heads (int): Number of query heads.
            num_kv_heads (int): Number of key/value heads.
            dropout (float): Dropout probability applied to attention weights.
            device (str): Torch device to place the model.
            dtype (torch.dtype): Data type used in the projections.
        """
        super().__init__()
        assert embed_dim % num_query_heads == 0, "embed_dim must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.device = device

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.kv_proj = nn.Linear(embed_dim, 2 * num_kv_heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)

        self.dropout = nn.Dropout(dropout)
        self._causal_mask_cache = {}

    def _get_or_create_causal_mask(self, seq_len):
        """
        Retrieves or creates a causal attention mask (upper triangular) to
        prevent attending to future tokens.

        Args:
            seq_len (int): Sequence length.

        Returns:
            torch.Tensor: A boolean mask of shape (seq_len, seq_len).
        """
        if seq_len not in self._causal_mask_cache:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device),
                diagonal=1
            )
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]

    def forward(self, x, mask=True):
        """
        Args:
            x (Tensor): Input tensor of shape (B, T, C).
            mask (bool): Whether to apply a causal mask.

        Returns:
            Tensor of shape (B, T, C) after applying grouped query attention.
        """
        B, T, C = x.shape
        x = x.to(device=self.device, dtype=self.q_proj.weight.dtype)

        # Project Q, K, V
        q = self.q_proj(x)  # (B, T, C)
        kv = self.kv_proj(x)  # (B, T, 2 * num_kv_heads * head_dim)
        k, v = kv.chunk(2, dim=-1)

        # Reshape projections
        q = q.view(B, T, self.num_query_heads, self.head_dim).transpose(1, 2)  # (B, num_query_heads, T, head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)     # (B, num_kv_heads, T, head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat keys and values for each query head group
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)  # (B, num_query_heads, T, head_dim)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Attention
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_query_heads, T, T)

        if mask:
            causal_mask = self._get_or_create_causal_mask(T)
            att = att.masked_fill(causal_mask[None, None, :, :], float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v  # (B, num_query_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)
    
class Linear_Attention(nn.Module):
    """
    Linear Attention module using kernel-based approximation for efficient attention.
    Implements a feature map-based linearized variant of self-attention using ELU+1 as the kernel.
    Inspired by "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention".

    Args:
        embed_dim (int): Input and output embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        eps (float): Small epsilon for numerical stability during division.
        device (str): Device to place the module on.
        dtype (torch.dtype): Tensor data type.
    """
    def __init__(self, embed_dim, num_heads, dropout, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.eps = eps
        self.device = device
        self.dtype = dtype

        # Linear projections for Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for linear attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, embed_dim)

        Returns:
            torch.Tensor: Output tensor of shape (B, T, embed_dim)
        """
        B, T, _ = x.shape

        # Compute Q, K, V and reshape for multi-head attention
        Q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        K = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)    # (B, H, T, D)
        V = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        # Feature map: phi(x) = ELU(x) + 1 (positive kernel trick)
        phi_q = F.elu(Q) + 1.0
        phi_k = F.elu(K) + 1.0

        # Cumulative KV (outer product and cumulative sum)
        kv_outer = torch.matmul(phi_k.unsqueeze(-1), V.unsqueeze(-2))  # (B, H, T, D, D)
        s = torch.cumsum(kv_outer, dim=2)                              # (B, H, T, D, D)
        z = torch.cumsum(phi_k, dim=2)                                 # (B, H, T, D)

        # Compute attention output
        numerator = torch.matmul(phi_q.unsqueeze(-2), s).squeeze(-2)   # (B, H, T, D)
        denominator = torch.sum(phi_q * z, dim=-1, keepdim=True) + self.eps  # (B, H, T, 1)
        out = numerator / denominator                                  # (B, H, T, D)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.dropout(self.out_proj(out))
    
class Multi_latent_Attention(nn.Module):
    """
    Multi-Latent Attention layer with separate learned compression for Q and KV streams.
    Useful for reducing computational/memory cost while preserving expressiveness.
    
    Args:
        embed_dim (int): Dimension of input/output embeddings.
        q_compressed_dim (int): Latent compression dimension for queries.
        kv_compressed_dim (int): Latent compression dimension for keys and values.
        num_heads (int): Number of attention heads.
        device (str): Device for tensors and modules.
        dtype (torch.dtype): Tensor type.
        dropout (float): Dropout probability.
    """
    def __init__(self, embed_dim, q_compressed_dim, kv_compressed_dim, num_heads,
                device='cpu', dtype=torch.float32, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.q_compressed_dim = q_compressed_dim
        self.kv_compressed_dim = kv_compressed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Query compression path
        self.W_dq = nn.Linear(embed_dim, q_compressed_dim, bias=False, device=device, dtype=dtype)
        self.W_dq_norm = nn.LayerNorm(q_compressed_dim, device=device, dtype=dtype)
        self.W_uq = nn.Linear(q_compressed_dim, embed_dim, bias=False, device=device, dtype=dtype)

        # Key/Value compression path
        self.W_dkv = nn.Linear(embed_dim, kv_compressed_dim, bias=False, device=device, dtype=dtype)
        self.W_dkv_norm = nn.LayerNorm(kv_compressed_dim, device=device, dtype=dtype)
        self.W_uk = nn.Linear(kv_compressed_dim, embed_dim, device=device, dtype=dtype)
        self.W_uv = nn.Linear(kv_compressed_dim, embed_dim, device=device, dtype=dtype)

        self.out_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, embed_dim)

        Returns:
            torch.Tensor: Output of shape (B, T, embed_dim)
        """
        B, T, _ = x.shape

        # Compress and reconstruct queries
        q_latent = self.W_dq_norm(self.W_dq(x))        # (B, T, q_compressed_dim)
        q_final = self.W_uq(q_latent)                  # (B, T, embed_dim)

        # Compress and reconstruct keys/values
        kv_latent = self.W_dkv_norm(self.W_dkv(x))     # (B, T, kv_compressed_dim)
        k_final = self.W_uk(kv_latent)                 # (B, T, embed_dim)
        v_final = self.W_uv(kv_latent)                 # (B, T, embed_dim)

        # Reshape for multi-head attention
        Q = q_final.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        K = k_final.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        V = v_final.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        # Scaled Dot-Product Attention (causal)
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            is_causal=True,
            dropout_p=self.dropout.p
        )  # (B, H, T, D)

        # Combine heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)  # (B, T, embed_dim)
        out = self.dropout(self.out_proj(out))                             # Final projection
        return out

class Local_Attention(nn.Module):
    """
    Local (Sliding Window) Multi-Head Self-Attention Layer.

    This module implements efficient **local attention** where each token attends only
    to a fixed-size window of previous tokens (including itself), enabling scalable
    attention for long sequences without full quadratic cost.

    Args:
        embed_dim (int): Total embedding dimension of the model.
        num_heads (int): Number of attention heads.
        window_size (int): Local attention window size. Each token attends to up to
            (window_size - 1) tokens before it.
        dropout (float): Dropout probability applied to attention weights.
        device (str): Device to run the module on (e.g., 'cpu' or 'cuda').
        dtype (torch.dtype): Data type for model parameters (e.g., torch.float32).

    Raises:
        AssertionError: If embed_dim is not divisible by num_heads.
        AssertionError: If window_size < 1 (would block all attention).

    Example:
        >>> attn = Local_Attention(embed_dim=256, num_heads=8, window_size=16)
        >>> x = torch.randn(2, 64, 256)  # (batch_size, seq_len, embed_dim)
        >>> out = attn(x)
        >>> out.shape  # torch.Size([2, 64, 256])
    """
    def __init__(self, embed_dim, num_heads, window_size, dropout=0.1, device='cpu', dtype=torch.float32):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert window_size >= 1, "Window size must be >= 1 to avoid full masking"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.device = device

        # QKV fused projection
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        # Cache causal masks for efficiency
        self._causal_mask_cache = {}

    def _get_or_create_sliding_window_mask(self, seq_len):
        """
        Creates or retrieves a cached local (sliding window) causal attention mask.

        For each position `i`, allows attention to tokens in range
        `[max(0, i - window_size + 1), i]`.

        Args:
            seq_len (int): Sequence length of the input.

        Returns:
            torch.BoolTensor: Shape (seq_len, seq_len). `True` indicates
            positions to be masked (not attended).
        """
        if seq_len not in self._causal_mask_cache:
            full_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device),
                diagonal=1
            )
            # Allow attention only within window
            band_mask = torch.triu(full_mask, diagonal=-(self.window_size - 1))
            self._causal_mask_cache[seq_len] = ~band_mask  # invert: True where disallowed
        return self._causal_mask_cache[seq_len]

    def forward(self, x, mask=True):
        """
        Forward pass for local (sliding window) multi-head self-attention.

        Args:
            x (torch.FloatTensor): Input tensor of shape (B, T, C), where
                B = batch size, T = sequence length, C = embedding dimension.
            mask (bool): If True, applies the local causal mask.

        Returns:
            torch.FloatTensor: Output tensor of shape (B, T, C).
        """
        B, T, C = x.shape
        x = x.to(device=self.device, dtype=self.qkv_proj.weight.dtype)

        # Fused QKV projection
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Split heads
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Apply sliding window causal mask
        if mask:
            causal_mask = self._get_or_create_sliding_window_mask(T)  # (T, T)
            att.masked_fill_(causal_mask[None, None, :, :], float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Attention-weighted sum
        out = att @ v  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class kv_cache_multihead(nn.Module):
    """
    Multi-Head Self-Attention with KV Caching and Rotary Position Embeddings (RoPE).

    This module supports autoregressive decoding by caching keys and values across forward passes.
    Ideal for GPT-style architectures.

    Features:
        - Fused QKV projection.
        - Rotary position embedding (RoPE).
        - Scaled dot-product attention with causal masking.
        - KV caching for fast inference.

    Args:
        embed_dim (int): Total embedding dimension.
        num_heads (int): Number of attention heads.
        batch_size (int): Fixed batch size for cache allocation.
        kv_seq_len (int): Maximum KV cache sequence length.
        dropout (float): Dropout on attention scores.
        device (str): Device to initialize weights and buffers on.
        dtype (torch.dtype): Precision type for parameters.

    Example:
        attn = KVCacheMultihead(embed_dim=256, num_heads=8, batch_size=4, kv_seq_len=1024)
        out = attn(x, start_pos=0)
        out.shape  # (4, 1024, 256)
    """
    def __init__(self, embed_dim, num_heads, batch_size, kv_seq_len, dropout=0.1, device='cpu', dtype=torch.float32):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.device = device
        self.dtype = dtype

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        # KV Cache (double kv_seq_len to avoid OOB)
        self.cache_keys = torch.zeros(batch_size, kv_seq_len * 2, num_heads, self.head_dim, device=device, dtype=dtype)
        self.cache_values = torch.zeros(batch_size, kv_seq_len * 2, num_heads, self.head_dim, device=device, dtype=dtype)

        self._causal_mask_cache = {}

    def _get_or_create_causal_mask(self, tgt_len: int, src_len: int):
        """
        Returns a [tgt_len, src_len] upper triangular causal mask (True = masked).
        """
        key = (tgt_len, src_len)
        if key not in self._causal_mask_cache:
            mask = torch.triu(torch.ones(tgt_len, src_len, dtype=torch.bool, device=self.device), diagonal=1)
            self._causal_mask_cache[key] = mask
        return self._causal_mask_cache[key]

    def _precompute_theta_position_frequency(self, head_dim: int, seq_len: int, theta: float = 10000.0):
        """
        Precomputes rotary position encodings as complex exponentials.

        Args:
            head_dim (int): Dimension of each attention head.
            seq_len (int): Sequence length.
            theta (float): Base for exponential frequency (default: 10000).

        Returns:
            torch.Tensor: Complex tensor of shape (seq_len, head_dim // 2)
        """
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        dim_half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim_half, device=self.device) / dim_half))
        pos = torch.arange(seq_len, device=self.device)
        freqs = torch.outer(pos, inv_freq)  # (seq_len, dim_half)

        freq_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freq_complex  # (seq_len, dim_half)

    def _apply_rotary_position_embedding(self, x: torch.Tensor, freq_complex: torch.Tensor):
        """
        Applies RoPE to input tensor using precomputed frequencies.

        Args:
            x (torch.Tensor): Input tensor of shape (B, num_heads, T, head_dim)
            freq_complex (torch.Tensor): Complex frequencies of shape (T, head_dim//2)

        Returns:
            torch.Tensor: Tensor with RoPE applied, same shape as input
        """
        B, H, T, D = x.shape
        assert D % 2 == 0, "head_dim must be even for RoPE"

        x = x.view(B, H, T, D // 2, 2)
        x_complex = torch.view_as_complex(x)  # (B, H, T, D//2)

        freq = freq_complex[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
        x_rot = x_complex * freq  # rotate via complex mult

        x_out = torch.view_as_real(x_rot).view(B, H, T, D)
        return x_out.to(dtype=self.dtype, device=self.device)

    def forward(self, x: torch.Tensor, start_pos: int, mask: bool = True, rope: bool = True):
        """
        Forward pass with optional RoPE and KV caching.

        Args:
            x (Tensor): Input of shape (B, T, C)
            start_pos (int): Start position for inserting into KV cache.
            mask (bool): Whether to apply causal mask.
            rope (bool): Whether to apply rotary position embeddings.

        Returns:
            Tensor: Output of shape (B, T, C)
        """
        B, T, C = x.shape
        assert C == self.embed_dim, "Input embed_dim mismatch"

        # Project input to Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to multi-head format
        q = q.view(B, T, self.num_heads, self.head_dim) # (B, T, H, D)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        if rope:
            freq = self._precompute_theta_position_frequency(self.head_dim, T)
            q = self._apply_rotary_position_embedding(q, freq)
            k = self._apply_rotary_position_embedding(k, freq)

        # Cache K and V
        end_pos = start_pos + T
        self.cache_keys[:B, start_pos:end_pos] = k.detach()
        self.cache_values[:B, start_pos:end_pos] = v.detach()

        # Use full cached KV up to end_pos
        k_full = self.cache_keys[:B, :end_pos].detach()  # (B, S, H, D)
        v_full = self.cache_values[:B, :end_pos].detach()

        q = q.transpose(1, 2) # (B, H, T, D)
        k_full = k_full.transpose(1, 2)  # (B, H, S, D)
        v_full = v_full.transpose(1, 2)  # (B, H, S, D)

        # Attention score computation
        attn_scores = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale  # (B, H, T, S)

        if mask:
            causal_mask = self._get_or_create_causal_mask(T, end_pos)  # (T, S)
            attn_scores = attn_scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        context = torch.matmul(attn_probs, v_full)  # (B, H, T, D)

        # Merge heads and project
        out = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class kv_cache_group_query(nn.Module):
    """
    Implements Grouped Query Attention (GQA) with rotary positional embeddings (RoPE)
    and KV caching for efficient autoregressive decoding.

    Args:
        embed_dim (int): Total embedding dimension.
        num_query_heads (int): Number of query heads.
        num_kv_heads (int): Number of key/value heads (shared across query heads).
        dropout (float): Dropout probability on attention weights.
        device (str): Target device for computation.
        dtype (torch.dtype): Data type used for model parameters and computation.
    """
    
    def __init__(self, embed_dim, num_query_heads, num_kv_heads, kv_seq_len, batch_size, dropout=0.1, device='cpu', dtype=torch.float32):
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

        # Linear projections: Q from full dim, KV from reduced dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.kv_proj = nn.Linear(embed_dim, 2 * num_kv_heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)

        self.dropout = nn.Dropout(dropout)
        self._causal_mask_cache = {}

        # KV Cache (double kv_seq_len to avoid OOB)
        self.cache_keys = torch.zeros(batch_size, kv_seq_len * 2, num_kv_heads, self.head_dim, device=device, dtype=dtype)
        self.cache_values = torch.zeros(batch_size, kv_seq_len * 2, num_kv_heads, self.head_dim, device=device, dtype=dtype)

    def _get_or_create_causal_mask(self, seq_len):
        """
        Returns a cached upper-triangular causal mask of shape (seq_len, seq_len).
        Prevents attending to future tokens.
        """
        if seq_len not in self._causal_mask_cache:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device),
                diagonal=1
            )
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]

    def _precompute_theta_position_frequency(self, head_dim: int, seq_len: int, theta: float = 10000.0):
        """
        Precomputes RoPE (rotary positional embedding) frequency matrix using complex numbers.
        """
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        dim_half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim_half, device=self.device) / dim_half))
        pos = torch.arange(seq_len, device=self.device)
        freqs = torch.outer(pos, inv_freq)  # (seq_len, dim_half)
        freq_complex = torch.polar(torch.ones_like(freqs), freqs)  # magnitude=1, angle=freqs
        return freq_complex  # (seq_len, dim_half)

    def _apply_rotary_position_embedding(self, x: torch.Tensor, freq_complex: torch.Tensor):
        """
        Applies RoPE to tensor `x` using precomputed complex frequencies.
        `x`: shape (B, H, T, D)
        """
        B, H, T, D = x.shape
        assert D % 2 == 0, "head_dim must be even for RoPE"

        x = x.view(B, H, T, D // 2, 2)  # Split last dim into real+imag
        x_complex = torch.view_as_complex(x)  # (B, H, T, D//2)
        freq = freq_complex[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
        x_rot = x_complex * freq  # Complex multiplication = rotation
        x_out = torch.view_as_real(x_rot).view(B, H, T, D)  # Back to real tensor
        return x_out.to(dtype=self.dtype, device=self.device)

    def forward(self, x, start_pos, mask=True, rope=True):
        """
        Forward pass for grouped-query attention with rotary positional embedding and KV cache.

        Args:
            x (Tensor): Input of shape (B, T, C)
            start_pos (int): Start position in KV cache for current chunk
            mask (bool): Whether to apply causal mask
            rope (bool): Whether to apply rotary position embedding
        Returns:
            Tensor of shape (B, T, C)
        """
        B, T, C = x.shape
        x = x.to(device=self.device, dtype=self.q_proj.weight.dtype)

        # Project input to Q, K, V
        q = self.q_proj(x)  # (B, T, C)
        kv = self.kv_proj(x)  # (B, T, 2 * num_kv_heads * head_dim)
        k, v = kv.chunk(2, dim=-1)  # (B, T, num_kv_heads * head_dim)

        # Reshape to multi-head form
        q = q.view(B, T, self.num_query_heads, self.head_dim) # (B, T, num_query_heads, head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim)  # (B, T, num_kv_heads, head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim)

        # Apply rotary positional embeddings
        if rope:
            freq_q = self._precompute_theta_position_frequency(self.head_dim, T)
            q = self._apply_rotary_position_embedding(q, freq_q)

            freq_k = self._precompute_theta_position_frequency(self.head_dim, self.kv_seq_len)
            k = self._apply_rotary_position_embedding(k, freq_k)

        # Write K/V to cache
        end_pos = start_pos + T
        self.cache_keys[:B, start_pos:end_pos] = k.detach() 
        self.cache_values[:B, start_pos:end_pos] = v.detach()

        # Read full K/V from cache up to current position
        k_full = self.cache_keys[:B, :end_pos].detach()
        v_full = self.cache_values[:B, :end_pos].detach()

        # Transpose Q: (B, T, H, D) -> (B, H, T, D)
        q = q.transpose(1, 2)
        # Transpose K/V: (B, S, kv_heads, D) -> (B, kv_heads, S, D)
        k_full = k_full.transpose(1, 2)
        v_full = v_full.transpose(1, 2)

        # Repeat KV heads to match Q heads
        k_full = k_full.repeat_interleave(self.num_queries_per_kv, dim=1)
        v_full = v_full.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Scaled dot-product attention
        att = (q @ k_full.transpose(-2, -1)) * self.scale  # (B, H, T, S)

        # Apply causal mask
        if mask:
            causal_mask = self._get_or_create_causal_mask(att.size(-1))
            att = att.masked_fill(causal_mask[None, None, -T:, :], float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Compute attention output
        out = att @ v_full  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)  # Final projection back to (B, T, C)