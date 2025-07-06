import torch
import torch.nn as nn
import torch.nn.functional as F

class Self_Attention(nn.Module):
    def __init__(self, emb_dim, dropout):
        super().__init__()
        self.key = nn.Linear(emb_dim, emb_dim, bias=False)
        self.query = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value = nn.Linear(emb_dim, emb_dim, bias=False)

        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: bool = True):
        batch_size, seq_len, emb_dim = x.size()

        q = self.query(x)  # (B, T, D)
        k = self.key(x)    # (B, T, D)
        v = self.value(x)  # (B, T, D)

        # Attention scores
        scores = q @ k.transpose(-2, -1) / (emb_dim ** 0.5)  # (B, T, T)

        if mask:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0)  # (1, T, T)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)  # (B, T, T)
        attn = self.dropout(attn)

        out = attn @ v  # (B, T, D)
        return self.out_proj(out)  # (B, T, D)

class Multi_Head_Attention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.key = nn.Linear(emb_dim, emb_dim, bias=False)
        self.query = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value = nn.Linear(emb_dim, emb_dim, bias=False)
        
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=True):
        B, T, _ = x.shape
        
        # Generate Q, K, V and reshape for multi-head attention
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, nh, T, T)

        # Apply causal mask if requested
        if mask:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # (B, nh, T, hd)
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, self.emb_dim)  # (B, T, emb_dim)
        
        return self.out_proj(out)
    
class Cross_MultiHead_Attention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        # Query, Key, Value projections
        self.query = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value = nn.Linear(emb_dim, emb_dim, bias=False)

        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        """
        x: (B, T_q, emb_dim) — query input (e.g., decoder hidden states)
        context: (B, T_kv, emb_dim) — source for keys/values (e.g., encoder output). If None, self-attention.
        mask: (B, 1, T_q, T_kv) — optional attention mask
        """
        B, T_q, _ = x.shape
        context = x if context is None else context  # self-attention fallback
        T_kv = context.shape[1]

        # Project Q, K, V
        q = self.query(x).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T_q, hd)
        k = self.key(context).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T_kv, hd)
        v = self.value(context).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T_kv, hd)

        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, nh, T_q, T_kv)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, nh, T_q, hd)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.emb_dim)  # (B, T_q, emb_dim)

        return self.out_proj(out)
    
class Multi_query_Attention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.query = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key = nn.Linear(emb_dim, self.head_dim, bias=False)
        self.value = nn.Linear(emb_dim, self.head_dim, bias=False)
        
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        
        # Generate Q, K, V and reshape for Multiquery_Attention
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.unsqueeze(1).expand(B, self.num_heads, T, self.head_dim)    # (B, 1, T, hd)
        v = v.unsqueeze(1).expand(B, self.num_heads, T, self.head_dim)
        
        # Compute attention scores
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, nh, T, T)

        # Apply causal mask if requested
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # (B, nh, T, hd)
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, self.emb_dim)  # (B, T, emb_dim)
        
        return self.out_proj(out)
    
class Group_query_Attention(nn.Module):
    def __init__(self, emb_dim, num_query_heads, num_kv_heads, dropout):
        super().__init__()
        assert emb_dim % num_query_heads == 0, "emb_dim must be divisible by num_heads"
        self.emb_dim = emb_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        
        self.head_dim = emb_dim // num_query_heads
        self.num_queries_pre_kv = num_query_heads // num_kv_heads

        self.query = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key = nn.Linear(emb_dim, self. num_kv_heads * self.head_dim, bias=False)
        self.value = nn.Linear(emb_dim, self.num_kv_heads * self.head_dim, bias=False)

        self.out_proj = nn.Linear(emb_dim, emb_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # Generate Q, K, V and reshape for Multiquery_Attention
        q = self.query(x).view(B, T, self.num_query_heads, self.head_dim).transpose(1, 2)  # (B, nqh, T, hd)
        k = self.key(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)     # (B, nkvh, T, hd)
        v = self.value(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)   # (B, nkvh, T, hd)

        k = k.repeat_interleave(self.num_queries_pre_kv,dim=1)
        v = v.repeat_interleave(self.num_queries_pre_kv,dim=1)

        # Compute attention scores
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, nh, T, T)

        # Apply causal mask if requested
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # (B, nh, T, hd)
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, self.emb_dim)  # (B, T, emb_dim)
        
        return self.out_proj(out)