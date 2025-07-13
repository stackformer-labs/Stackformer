import torch
import torch.nn as nn
import torch.nn.functional as F

class Self_Attention(nn.Module):
    def __init__(self, Emb_dim, dropout,dtype=torch.float32,device='cpu'):
        super().__init__()
        self.scale = torch.tensor(Emb_dim ** 0.5,dtype=dtype,device=device)
        
        self.key = nn.Linear(Emb_dim, Emb_dim, bias=False,dtype=dtype,device=device)
        self.query = nn.Linear(Emb_dim, Emb_dim, bias=False,dtype=dtype,device=device)
        self.value = nn.Linear(Emb_dim, Emb_dim, bias=False,dtype=dtype,device=device)

        self.out_proj = nn.Linear(Emb_dim, Emb_dim,dtype=dtype,device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: bool = True):
        Batch_size, Seq_len, Emb_dim = x.size()
        
        Querys = self.query(x)  # (Batch_size, Seq_len, D)
        Keys = self.key(x)    # (Batch_size, Seq_len, D)
        Values = self.value(x)  # (Batch_size, Seq_len, D)

        # Attention scores
        scores = Querys @ Keys.transpose(-2, -1) / self.scale  # (Batch_size, Seq_len, Seq_len)

        if mask:
            causal_mask = torch.triu(torch.ones(Seq_len, Seq_len, device=x.device)).unsqueeze(0)  # (1, Seq_len, Seq_len)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)  # (Batch_size, Seq_len, Seq_len)
        attn = self.dropout(attn)

        out = (attn @ Values) # (Batch_size, Seq_len, D)
        return self.out_proj(out)  # (Batch_size, Seq_len, D)

class Multi_Head_Attention(nn.Module):
    def __init__(self, Emb_dim, num_heads, dropout, device='cpu',dtype=torch.float32):
        super().__init__()
        assert Emb_dim % num_heads == 0, "Emb_dim must be divisible by num_heads"
        self.Emb_dim = Emb_dim
        self.num_heads = num_heads
        self.head_dim = Emb_dim // num_heads

        self.key = nn.Linear(Emb_dim, Emb_dim, bias=False,dtype=dtype,device=device)
        self.query = nn.Linear(Emb_dim, Emb_dim, bias=False,dtype=dtype,device=device)
        self.value = nn.Linear(Emb_dim, Emb_dim, bias=False,dtype=dtype,device=device)
        
        self.scale = torch.tensor(self.head_dim ** 0.5,device=device,dtype=dtype)
        self.out_proj = nn.Linear(Emb_dim, Emb_dim,dtype=dtype,device=device)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=True):
        Batch_size, Seq_len, _ = x.shape
        
        # Generate Q, K, V and reshape for multi-head attention
        Keys = self.key(x).view(Batch_size, Seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (Batch_size, nh, Seq_len, hd)
        Querys = self.query(x).view(Batch_size, Seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        Values = self.value(x).view(Batch_size, Seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = (Querys @ Keys.transpose(-2, -1)) / self.scale  # (Batch_size, nh, Seq_len, Seq_len)

        # Apply causal mask if requested
        if mask:
            causal_mask = torch.triu(torch.ones(Seq_len, Seq_len, device=x.device)).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ Values  # (Batch_size, nh, Seq_len, hd)
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(Batch_size, Seq_len, self.Emb_dim)  # (Batch_size, Seq_len, Emb_dim)
        
        return self.out_proj(out)
    
class Cross_MultiHead_Attention(nn.Module):
    def __init__(self, Emb_dim, num_heads, dropout,device='cpu', dtype=torch.float32):
        super().__init__()
        assert Emb_dim % num_heads == 0, "Emb_dim must be divisible by num_heads"
        self.Emb_dim = Emb_dim
        self.num_heads = num_heads
        self.head_dim = Emb_dim // num_heads

        # Querys, Key, Value projections
        self.query = nn.Linear(Emb_dim, Emb_dim, bias=False,dtype=dtype,device=device)
        self.key = nn.Linear(Emb_dim, Emb_dim, bias=False,dtype=dtype,device=device)
        self.value = nn.Linear(Emb_dim, Emb_dim, bias=False,dtype=dtype,device=device)

        self.scale = torch.tensor(self.head_dim ** 0.5,device=device,dtype=dtype)

        self.out_proj = nn.Linear(Emb_dim, Emb_dim,dtype=dtype,device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        """
        x: (Batch_size, query_seq_len, Emb_dim) — query input (e.g., decoder hidden states)
        context: (Batch_size, KV_seq_len, Emb_dim) — source for keys/values (e.g., encoder output). If None, self-attention.
        mask: (Batch_size, 1, query_seq_len, KV_seq_len) — optional attention mask
        """
        Batch_size, query_seq_len, _ = x.shape
        context = x if context is None else context  # self-attention fallback
        KV_seq_len = context.shape[1]

        # Project Q, K, V
        Querys = self.query(x).view(Batch_size, query_seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (Batch_size, nh, query_seq_len, hd)
        Keys = self.key(context).view(Batch_size, KV_seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (Batch_size, nh, KV_seq_len, hd)
        Values = self.value(context).view(Batch_size, KV_seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (Batch_size, nh, KV_seq_len, hd)

        # Attention scores
        scores = (Querys @ Keys.transpose(-2, -1)) / self.scale  # (Batch_size, nh, query_seq_len, KV_seq_len)

        if mask is not None:            
            causal_mask = torch.triu(torch.ones(query_seq_len, query_seq_len, device=x.device)).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ Values  # (Batch_size, nh, query_seq_len, hd)
        out = out.transpose(1, 2).contiguous().view(Batch_size, query_seq_len, self.Emb_dim)  # (Batch_size, query_seq_len, Emb_dim)

        return self.out_proj(out)
    
class Multi_query_Attention(nn.Module):
    def __init__(self, Emb_dim, num_heads, dropout, device='cpu', dtype=torch.float32):
        super().__init__()
        assert Emb_dim % num_heads == 0, "Emb_dim must be divisible by num_heads"
        self.Emb_dim = Emb_dim
        self.num_heads = num_heads
        self.head_dim = Emb_dim // num_heads

        self.query = nn.Linear(Emb_dim, Emb_dim, bias=False,dtype=dtype,device=device)
        self.key = nn.Linear(Emb_dim, self.head_dim, bias=False,dtype=dtype,device=device)
        self.value = nn.Linear(Emb_dim, self.head_dim, bias=False,dtype=dtype,device=device)
        
        self.scale = torch.tensor(self.head_dim ** 0.5,device=device,dtype=dtype)
        
        self.out_proj = nn.Linear(Emb_dim, Emb_dim,dtype=dtype,device=device)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Batch_size, Seq_len, C = x.shape
        # Generate Q, K, V and reshape for Multiquery_Attention
        Querys = self.query(x)
        Keys = self.key(x)
        Values = self.value(x)
        
        Querys = Querys.view(Batch_size, Seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (Batch_size, nh, Seq_len, hd)
        Keys = Keys.unsqueeze(1).expand(Batch_size, 1, Seq_len, self.head_dim)    # (Batch_size, 1, Seq_len, hd)
        Values = Values.unsqueeze(1).expand(Batch_size, 1, Seq_len, self.head_dim)
        
        # Compute attention scores
        scores = (Querys @ Keys.transpose(-2, -1)) / self.scale  # (Batch_size, nh, Seq_len, Seq_len)

        # Apply causal mask if requested
        causal_mask = torch.triu(torch.ones(Seq_len, Seq_len, device=x.device)).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ Values  # (Batch_size, nh, Seq_len, hd)
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(Batch_size, Seq_len, self.Emb_dim)  # (Batch_size, Seq_len, Emb_dim)
        
        return self.out_proj(out)
    
class Group_query_Attention(nn.Module):
    def __init__(self, Emb_dim, num_query_heads, num_kv_heads, dropout,device='cpu', dtype=torch.float32):
        super().__init__()
        assert Emb_dim % num_query_heads == 0, "Emb_dim must be divisible by num_heads"
        self.Emb_dim = Emb_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        
        self.head_dim = Emb_dim // num_query_heads
        self.num_queries_pre_kv = num_query_heads // num_kv_heads

        self.query = nn.Linear(Emb_dim, Emb_dim, bias=False,dtype=dtype,device=device)
        self.key = nn.Linear(Emb_dim, self. num_kv_heads * self.head_dim, bias=False,dtype=dtype,device=device)
        self.value = nn.Linear(Emb_dim, self.num_kv_heads * self.head_dim, bias=False,dtype=dtype,device=device)
        
        self.scale = torch.tensor(self.head_dim ** 0.5,device=device,dtype=dtype)

        self.out_proj = nn.Linear(Emb_dim, Emb_dim,dtype=dtype,device=device)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Batch_size, Seq_len, C = x.shape

        # Generate Q, K, V and reshape for Multiquery_Attention
        Querys = self.query(x).view(Batch_size, Seq_len, self.num_query_heads, self.head_dim).transpose(1, 2)  # (Batch_size, nqh, Seq_len, hd)
        Keys = self.key(x).view(Batch_size, Seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)     # (Batch_size, nkvh, Seq_len, hd)
        Values = self.value(x).view(Batch_size, Seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)   # (Batch_size, nkvh, Seq_len, hd)

        Keys = Keys.repeat_interleave(self.num_queries_pre_kv,dim=1)
        Values = Values.repeat_interleave(self.num_queries_pre_kv,dim=1)

        # Compute attention scores
        scores = (Querys @ Keys.transpose(-2, -1)) / self.scale  # (Batch_size, nh, Seq_len, Seq_len)

        # Apply causal mask if requested
        causal_mask = torch.triu(torch.ones(Seq_len, Seq_len, device=x.device)).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ Values  # (Batch_size, nh, Seq_len, hd)
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(Batch_size, Seq_len, self.Emb_dim)  # (Batch_size, Seq_len, Emb_dim)
        
        return self.out_proj(out)
    
class Linear_Attention(nn.Module):
    def __init__(self, Emb_dim, num_heads, dropout, eps = 1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        assert Emb_dim % num_heads == 0, "Emb_dim must be divisible by num_heads"
        self.Emb_dim = Emb_dim
        self.eps = eps
        self.dtype=dtype
        self.num_heads = num_heads
        self.head_dim = Emb_dim // self.num_heads

        self.query = nn.Linear(Emb_dim, Emb_dim, bias=False,dtype=dtype,device=device)
        self.key = nn.Linear(Emb_dim, Emb_dim, bias=False,dtype=dtype,device=device)
        self.value = nn.Linear(Emb_dim, Emb_dim, bias=False,dtype=dtype,device=device)
        
        self.out_proj = nn.Linear(Emb_dim, Emb_dim,dtype=dtype,device=device)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Batch_size, Seq_len, _ = x.shape

        # Generate Q, K, V and reshape for multi-head attention
        Querys = self.query(x).view(Batch_size, Seq_len, self.num_heads, self.head_dim).transpose(1, 2)     # (Batch_size, nh, Seq_len, hd)
        Keys = self.key(x).view(Batch_size, Seq_len, self.num_heads, self.head_dim).transpose(1, 2)  
        Values = self.value(x).view(Batch_size, Seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        phi_q = F.elu(Querys) + 1.0
        phi_k = F.elu(Keys) + 1.0
        
        kv_outer_product = torch.matmul(phi_k.unsqueeze(-1),Values.unsqueeze(-2))
        
        s_cumulative = torch.cumsum(kv_outer_product, dim=2)
        z_cumulative = torch.cumsum(phi_k,dim=2)
        
        numerator = torch.matmul(phi_q.unsqueeze(-2),s_cumulative).squeeze(-2)
        denominator = torch.sum(phi_q * z_cumulative,dim=-1,keepdim=True) + self.eps
        
        out = numerator / denominator
        out = out.transpose(1, 2).contiguous().view(Batch_size, Seq_len, self.Emb_dim)  # (Batch_size, Seq_len, Emb_dim)
        out = self.out_proj(out)
        return self.dropout(out)
    
class Multi_latent_Attention(nn.Module):
    def __init__(self, Emb_dim, q_compressed_dim, kv_compressed_dim , num_heads,device='cpu' ,dtype=torch.float32, dropout=0):
        super().__init__()
        assert Emb_dim % num_heads == 0, "Emb_dim must be divisible by num_heads"
        self.Emb_dim = Emb_dim
        self.q_compressed_dim = q_compressed_dim
        self.kv_compressed_dim = kv_compressed_dim
        self.num_heads = num_heads
        self.head_dim = Emb_dim // self.num_heads

        self.W_dq = nn.Linear(Emb_dim,q_compressed_dim,bias=False,dtype=dtype,device=device)
        self.W_dq_norm = nn.LayerNorm(q_compressed_dim,dtype=dtype,device=device)
        self.W_uq = nn.Linear(q_compressed_dim,Emb_dim,bias=False,dtype=dtype,device=device)

        self.W_dkv = nn.Linear(Emb_dim,kv_compressed_dim,bias=False,dtype=dtype,device=device)
        self.W_dkv_norm = nn.LayerNorm(kv_compressed_dim,dtype=dtype,device=device)
        self.W_uk = nn.Linear(kv_compressed_dim,Emb_dim,dtype=dtype,device=device)
        self.W_uv = nn.Linear(kv_compressed_dim,Emb_dim,dtype=dtype,device=device)

        self.out_proj = nn.Linear(Emb_dim, Emb_dim,dtype=dtype,device=device)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Batch_size, Seq_len, C = x.shape

        compressed_q_latent = self.W_dq(x)
        compressed_q_latent_norm = self.W_dq_norm(compressed_q_latent)
        q_final = self.W_uq(compressed_q_latent_norm)

        compressed_kv_latent = self.W_dkv(x)
        compressed_kv_latent_norm = self.W_dkv_norm(compressed_kv_latent)
        k_final = self.W_uk(compressed_kv_latent_norm)
        v_final = self.W_uv(compressed_kv_latent_norm)

        Querys = q_final.view(Batch_size, Seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        Keys = k_final.view(Batch_size, Seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        Values = v_final.view(Batch_size, Seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            query=Querys,
            key=Keys,
            value=Values,
            attn_mask=None,
            is_causal=True,
            dropout_p=self.dropout.p  # use self.dropout.p to get dropout prob
        )
        out = self.out_proj(out.transpose(1, 2).contiguous().view(Batch_size, Seq_len, C))
        out = self.dropout(out)
        return out

def precompute_theta_position_frequency(head_dim, seq_len, device='cpu', theta=10000.0):
    assert head_dim % 2 == 0, "head_dim must be even"

    # Frequencies: 1 / (theta ** (2i / head_dim))
    theta_numerator = torch.arange(0, head_dim, 2, device=device)
    inv_freq = 1.0 / (theta ** (theta_numerator / head_dim))

    # Position indices
    m = torch.arange(seq_len, device=device)

    # Outer product: (seq_len, head_dim // 2)
    freqs = torch.outer(m, inv_freq)

    # Convert to complex exponential form: exp(i * freq)
    freq_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freq_complex


def apply_rotry_position_embedding(x, freq_complex, device='cpu', dtype=torch.float32):
    # x: (batch_size, seq_len, num_head, emb_dim)
    batch_size, seq_len, num_head, emb_dim = x.shape
    assert emb_dim % 2 == 0, "emb_dim must be even"

    # Reshape to split last dimension into complex pairs
    x_reshaped = x.view(batch_size, seq_len, num_head, emb_dim // 2, 2).to(device=device, dtype=dtype)
    x_complex = torch.view_as_complex(x_reshaped)

    # Prepare frequencies: (1, seq_len, 1, emb_dim//2)
    freq_complex = freq_complex[:seq_len].unsqueeze(0).unsqueeze(2).to(device=device)

    # Apply rotation
    x_rotated = x_complex * freq_complex

    # Convert back to real tensor and reshape
    x_out = torch.view_as_real(x_rotated).contiguous().view(batch_size, seq_len, num_head, emb_dim)
    return x_out.to(device=device, dtype=dtype)

class kv_cache_multihead(nn.Module):
    def __init__(self, emb_dim, num_heads, batch_size, kv_seq_len, device='cpu', dtype=torch.float32,dropout=0.1):
        super().__init__()        
        self.dtype = dtype
        self.device = device
        
        assert emb_dim % num_heads == 0
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.kv_seq_len = kv_seq_len

        self.query = nn.Linear(emb_dim, emb_dim, bias=False,dtype=dtype,device=device)
        self.key = nn.Linear(emb_dim, emb_dim, bias=False,dtype=dtype,device=device)
        self.value = nn.Linear(emb_dim, emb_dim, bias=False,dtype=dtype,device=device)
        
        self.out_proj = nn.Linear(emb_dim, emb_dim,dtype=dtype,device=device)
        self.dropout = nn.Dropout(dropout)

        self.cache_keys = torch.zeros(batch_size, kv_seq_len, num_heads, self.head_dim,dtype=dtype,device=device)
        self.cache_value = torch.zeros(batch_size, kv_seq_len, num_heads, self.head_dim,dtype=dtype,device=device)

    def forward(self, x, start_pos, RoPE: False):
        batch_size, seq_len, C = x.shape
        
        xq = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        xv = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        if RoPE:
            freq_complex = precompute_theta_position_frequency(head_dim=self.head_dim, seq_len=seq_len, device=self.device)
            xq = apply_rotry_position_embedding(xq, freq_complex, device=self.device, dtype=self.dtype)
            freq_complex = precompute_theta_position_frequency(head_dim=self.head_dim, seq_len=self.kv_seq_len, device=self.device)
            xk = apply_rotry_position_embedding(xk, freq_complex, device=self.device, dtype=self.dtype)
        
        # Cache keys and values
        self.cache_keys[:, start_pos:start_pos+seq_len] = xk
        self.cache_value[:, start_pos:start_pos+seq_len] = xv

        xk_full = self.cache_keys[:, :start_pos+seq_len]
        xv_full = self.cache_value[:, :start_pos+seq_len]

        query = xq.transpose(1, 2)         # (batch_size, num_head, seq_len, emb_dim)
        key = xk_full.transpose(1, 2)    # (batch_size, num_head, T_total, emb_dim)
        value = xv_full.transpose(1, 2)    # (batch_size, num_head, T_total, emb_dim)
                
        attn_scores = torch.matmul(query, key.transpose(2, 3)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, value)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.dropout(self.out_proj(out))

class kv_cache_group_query(nn.Module):
    def __init__(self, emb_dim, query_num_heads, kv_num_heads, batch_size, kv_seq_len,device='cpu' , dtype=torch.float32 , dropout=0.1):
        super().__init__()
        self.dtype = dtype
        self.device = device
        
        assert query_num_heads % kv_num_heads == 0, "query heads must be divisible by kv heads"
        assert emb_dim % query_num_heads == 0, "embedding must be divisible by query heads"

        self.emb_dim = emb_dim
        self.query_num_heads = query_num_heads
        self.kv_num_heads = kv_num_heads
        self.head_dim = emb_dim // query_num_heads
        self.num_queries_per_kv = query_num_heads // kv_num_heads
        self.kv_seq_len = kv_seq_len

        self.query = nn.Linear(emb_dim, emb_dim, bias=False,dtype=dtype,device=device)
        self.key = nn.Linear(emb_dim, kv_num_heads * self.head_dim, bias=False,dtype=dtype,device=device)
        self.value = nn.Linear(emb_dim, kv_num_heads * self.head_dim, bias=False,dtype=dtype,device=device)

        self.out_proj = nn.Linear(emb_dim, emb_dim,dtype=dtype,device=device)
        self.dropout = nn.Dropout(dropout)

        # KV caches
        self.register_buffer("cache_keys", torch.zeros(batch_size, kv_seq_len, kv_num_heads, self.head_dim,device=device,dtype=dtype))
        self.register_buffer("cache_value", torch.zeros(batch_size, kv_seq_len, kv_num_heads, self.head_dim,device=device,dtype=dtype))
        
    def forward(self, x, start_pos, RoPE=False):
        batch_size, seq_len, _ = x.shape

        xq = self.query(x).view(batch_size, seq_len, self.query_num_heads, self.head_dim)
        xk = self.key(x).view(batch_size, seq_len, self.kv_num_heads, self.head_dim)
        xv = self.value(x).view(batch_size, seq_len, self.kv_num_heads, self.head_dim)

        if RoPE:
            freq_q = precompute_theta_position_frequency(head_dim=self.head_dim, seq_len=seq_len, device=self.device)
            xq = apply_rotry_position_embedding(xq, freq_q, device=self.device, dtype=self.dtype)
            freq_k = precompute_theta_position_frequency(head_dim=self.head_dim, seq_len=self.kv_seq_len, device=self.device)
            xk = apply_rotry_position_embedding(xk, freq_k, device=self.device, dtype=self.dtype)
        # Cache
        self.cache_keys[:, start_pos:start_pos+seq_len] = xk
        self.cache_value[:, start_pos:start_pos+seq_len] = xv

        xk_full = self.cache_keys[:, :start_pos+seq_len]  # [B, T, kv_heads, D]
        xv_full = self.cache_value[:, :start_pos+seq_len]

        # Transpose for attention: [B, H, T, D]
        query = xq.transpose(1, 2)  # [B, q_heads, seq_len, D]
        key = xk_full.transpose(1, 2)  # [B, kv_heads, total_kv_len, D]
        value = xv_full.transpose(1, 2)
        
        # Repeat keys and values to match query heads
        key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
        value = value.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Attention
        attn_scores = torch.matmul(query, key.transpose(2, 3)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, value)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.emb_dim)
        return self.dropout(self.out_proj(out))