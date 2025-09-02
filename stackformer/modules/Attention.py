import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Self_Attention(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, device='cpu', dtype=torch.float32):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.dtype = dtype

        # Scaling factor for dot-product attention
        self.scale = 1.0 / math.sqrt(embed_dim)

        # Linear layer
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        
        # Output projection layer to map attention output back to input dimension
        self.out_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)

        # Dropout applied to the attention weights
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

        # Cache for causal masks to avoid recomputation across forward passes
        self._causal_mask_cache = {}

    def _get_or_create_causal_mask(self, seq_len):
        if seq_len not in self._causal_mask_cache:
            # Upper triangular mask with True above the diagonal
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device),
                diagonal=1
            )
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]

    def forward(self, x, mask=True):
        B, T, C = x.shape
        x = x.to(device=self.device, dtype=self.dtype)
        
        # Compute queries, keys, and values
        q = self.q_proj(x)  # Shape: (B, T, C)
        k = self.k_proj(x)
        v = self.v_proj(x)

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
    def __init__(self, embed_dim, num_heads, dropout=0.1, device='cpu', dtype=torch.float32):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Each head gets a slice of the embedding
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.device = device
        self.dtype = dtype

        # Linear layer
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=self.dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=self.dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=self.dtype)
        
        # Final output projection after attention
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=self.dtype)
        
        self.dropout = nn.Dropout(dropout)

        # Cache for causal masks keyed by sequence length
        self._causal_mask_cache = {}

    def _get_or_create_causal_mask(self, seq_len):
        if seq_len not in self._causal_mask_cache:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device),
                diagonal=1
            )
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]

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

class Cross_MultiHead_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, device='cpu', dtype=torch.float32):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.device = device
        self.dtype = dtype

        # Linear layer
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        # Cache for causal masks (optional)
        self._causal_mask_cache = {}

    def _get_or_create_causal_mask(self, seq_len):
        if seq_len not in self._causal_mask_cache:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device),
                diagonal=1
            )
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]

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
    def __init__(self, embed_dim, num_heads, dropout=0.1, device='cpu', dtype=torch.float32):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.device = device
        self.dtype = dtype

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=self.dtype)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=False, device=device, dtype=self.dtype)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=False, device=device, dtype=self.dtype)
        
        # Output final projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=self.dtype)

        self.dropout = nn.Dropout(dropout)

        # Cache for causal masks (for autoregressive models)
        self._causal_mask_cache = {}

    def _get_or_create_causal_mask(self, seq_len):
        if seq_len not in self._causal_mask_cache:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device),
                diagonal=1
            )
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]

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
    def __init__(self, embed_dim, num_query_heads, num_kv_heads, dropout=0.1, device='cpu', dtype=torch.float32):
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
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)

        self.dropout = nn.Dropout(dropout)
        self._causal_mask_cache = {}

    def _get_or_create_causal_mask(self, seq_len):
        if seq_len not in self._causal_mask_cache:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device),
                diagonal=1
            )
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]

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
    
class Local_Attention(nn.Module):
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
        self.dtype = dtype

        # QKV projection
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=self.dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=self.dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=self.dtype)
        
        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=self.dtype)
        self.dropout = nn.Dropout(dropout)

        # Cache causal masks for efficiency
        self._causal_mask_cache = {}

    def _get_or_create_sliding_window_mask(self, seq_len):
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
    def __init__(self, embed_dim, num_heads, batch_size, kv_seq_len, dropout=0.1, device='cpu', dtype=torch.float32):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.device = device
        self.dtype = dtype
        
        #  QKV projection
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        
        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        # KV Cache (double kv_seq_len to avoid OOB)
        self.cache_keys = torch.zeros(batch_size, kv_seq_len * 2, num_heads, self.head_dim, device=device, dtype=dtype)
        self.cache_values = torch.zeros(batch_size, kv_seq_len * 2, num_heads, self.head_dim, device=device, dtype=dtype)

        self._causal_mask_cache = {}

    def _get_or_create_causal_mask(self, tgt_len: int, src_len: int):
        key = (tgt_len, src_len)
        if key not in self._causal_mask_cache:
            mask = torch.triu(torch.ones(tgt_len, src_len, dtype=torch.bool, device=self.device), diagonal=1)
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

    def forward(self, x: torch.Tensor, start_pos: int, mask: bool = True, rope: bool = True):
        B, T, C = x.shape
        assert C == self.embed_dim, "Input embed_dim mismatch"

        # Project input to Q, K, V
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, C)
        v = self.v_proj(x)  # (B, T, C)

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
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)

        self.dropout = nn.Dropout(dropout)
        self._causal_mask_cache = {}

        # KV Cache (double kv_seq_len to avoid OOB)
        self.cache_keys = torch.zeros(batch_size, kv_seq_len * 2, num_kv_heads, self.head_dim, device=device, dtype=dtype)
        self.cache_values = torch.zeros(batch_size, kv_seq_len * 2, num_kv_heads, self.head_dim, device=device, dtype=dtype)

    def _get_or_create_causal_mask(self, seq_len):
        if seq_len not in self._causal_mask_cache:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device),
                diagonal=1
            )
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]

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

        # Project input to Q, K, V
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, num_kv_heads * head_dim)
        v = self.v_proj(x)

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