import math
import torch
import torch.nn as nn
import torch.nn.functional as F 

# --- position embedding ---
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, seq_len, emb_dim):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim

        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))

        pe = torch.zeros(seq_len, emb_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, emb_dim) or (batch_size, seq_len)
        batch_size, seq_len = x.shape[0], x.shape[1]
        out = self.pe[:seq_len].unsqueeze(0).expand(batch_size, seq_len, -1)
        return out.to(device=x.device,dtype=x.dtype)

# --- multi-head attention ---
class Multi_Head_Attention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, device='cpu',dtype=torch.float32):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.device = device
        self.head_dim = emb_dim // num_heads

        self.key = nn.Linear(emb_dim, emb_dim, bias=False,dtype=dtype,device=device)
        self.query = nn.Linear(emb_dim, emb_dim, bias=False,dtype=dtype,device=device)
        self.value = nn.Linear(emb_dim, emb_dim, bias=False,dtype=dtype,device=device)
        
        self.scale = torch.tensor(self.head_dim ** 0.5,device=device,dtype=dtype)
        self.out_proj = nn.Linear(emb_dim, emb_dim,dtype=dtype,device=device)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Batch_size, Seq_len, _ = x.shape
        
        # Generate Q, K, V and reshape for multi-head attention
        Keys = self.key(x).view(Batch_size, Seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (Batch_size, nh, Seq_len, hd)
        Querys = self.query(x).view(Batch_size, Seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        Values = self.value(x).view(Batch_size, Seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = (Querys @ Keys.transpose(-2, -1)) / self.scale  # (Batch_size, nh, Seq_len, Seq_len)

        # Apply causal mask if requested
        causal_mask = torch.triu(torch.ones(Seq_len, Seq_len, dtype=torch.bool, device=self.device), diagonal=1)
        scores = scores.masked_fill_(causal_mask[None, None, :, :], float('-inf'))
        
        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ Values  # (Batch_size, nh, Seq_len, hd)
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(Batch_size, Seq_len, self.emb_dim)  # (Batch_size, Seq_len, emb_dim)
        
        return self.out_proj(out)
    
# --- cross-attention ---
class Cross_MultiHead_Attention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout,device='cpu', dtype=torch.float32):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.emb_dim = emb_dim
        self.device = device
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        # Querys, Key, Value projections
        self.query = nn.Linear(emb_dim, emb_dim, bias=False,dtype=dtype,device=device)
        self.key = nn.Linear(emb_dim, emb_dim, bias=False,dtype=dtype,device=device)
        self.value = nn.Linear(emb_dim, emb_dim, bias=False,dtype=dtype,device=device)

        self.scale = torch.tensor(self.head_dim ** 0.5,device=device,dtype=dtype)

        self.out_proj = nn.Linear(emb_dim, emb_dim,dtype=dtype,device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        Batch_size, query_seq_len, _ = x.shape
        context = x if context is None else context  # self-attention fallback
        KV_seq_len = context.shape[1]

        # Project Q, K, V
        Querys = self.query(x).view(Batch_size, query_seq_len, self.num_heads, self.head_dim).transpose(1, 2)  
        Keys = self.key(context).view(Batch_size, KV_seq_len, self.num_heads, self.head_dim).transpose(1, 2)  
        Values = self.value(context).view(Batch_size, KV_seq_len, self.num_heads, self.head_dim).transpose(1, 2)  

        # Attention scores
        scores = (Querys @ Keys.transpose(-2, -1)) / self.scale  # (Batch_size, nh, query_seq_len, KV_seq_len)

        causal_mask = torch.triu(torch.ones(query_seq_len, query_seq_len, dtype=torch.bool, device=self.device), diagonal=1)
        scores = scores.masked_fill_(causal_mask[None, None, :, :], float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ Values  
        out = out.transpose(1, 2).contiguous().view(Batch_size, query_seq_len, self.emb_dim)  # (Batch_size, query_seq_len, emb_dim)

        return self.out_proj(out)
    
# --- Feed Forward ---
class FF_ReLU(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout=0.1, device='cpu', dtype=torch.float32):
        super().__init__()
        self.relu = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim, device=device, dtype=dtype),
        )
    
    def forward(self, x):
        return self.relu(x)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(emb_dim, device=device, dtype=dtype))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return norm_x * self.weight + self.bias
    
class Encoder(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.attention = Multi_Head_Attention(emb_dim, num_heads, dropout, device=device, dtype=dtype)
        self.norm1 = LayerNorm(emb_dim, eps=eps, device=device, dtype=dtype)
        self.ff_relu = FF_ReLU(emb_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm2 = LayerNorm(emb_dim, eps=eps, device=device, dtype=dtype)
        
    def forward(self, x):
        residual = x
        x = self.attention(x)
        x = self.norm1(x)
        x = x + residual 
        
        residual = x
        x = self.ff_relu(x)
        x = self.norm2(x)
        x = x + residual  
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.attention = Multi_Head_Attention(emb_dim, num_heads, dropout, device=device, dtype=dtype)
        self.norm1 = LayerNorm(emb_dim, eps=eps, device=device, dtype=dtype)
        self.cross_attention = Cross_MultiHead_Attention(emb_dim, num_heads, dropout, device=device, dtype=dtype)
        self.norm2 = LayerNorm(emb_dim, eps=eps, device=device, dtype=dtype)
        self.ff_relu = FF_ReLU(emb_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm3 = LayerNorm(emb_dim, eps=eps, device=device, dtype=dtype)
        
    def forward(self, x, enc_output):
        residual = x
        x = self.attention(x)
        x = self.norm1(x)
        x = x + residual 
        
        residual = x
        x = self.cross_attention(x, context = enc_output)
        x = self.norm2(x)
        x = x + residual
        
        residual = x
        x = self.ff_relu(x)
        x = self.norm3(x)
        x = x + residual  
        
        return x
        
class transformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_heads, dropout, hidden_dim, 
                encoder_layers, decoder_layers, seq_len, eps=1e-5, device='cpu', dtype=torch.float32,
                ):
        super().__init__()
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        
        self.token_emb = nn.Embedding(vocab_size, emb_dim, device=device, dtype=dtype)
        self.pos = SinusoidalPositionalEmbedding(seq_len=seq_len, emb_dim=emb_dim)
        
        self.encoder_stack = nn.ModuleList([
            Encoder(emb_dim, num_heads, dropout, hidden_dim, eps=eps, device=device, dtype=dtype)
            for _ in range(encoder_layers)
        ])
        
        self.decoder_stack = nn.ModuleList([
            Decoder(emb_dim, num_heads, dropout, hidden_dim, eps=eps, device=device, dtype=dtype)
            for _ in range(decoder_layers)
        ])
        
        # --- final norm ---
        self.final_norm = LayerNorm(emb_dim, eps=eps, device=device, dtype=dtype)
        
        # --- output projection ---
        self.out_proj = nn.Linear(emb_dim, vocab_size, device=device, dtype=dtype)
        
    def encoder(self, x):
        x = self.token_emb(x) + self.pos(x)
        for block in self.encoder_stack:
            x = block(x)
        return x
    
    def decoder(self, x, enc_output):
        x = self.token_emb(x) + self.pos(x)
        for block in self.decoder_stack:
            x = block(x, enc_output)
        return x
    
    def forward(self, source, target):
        enc_output = self.encoder(source)
        out = self.decoder(target, enc_output)
        out = self.final_norm(out)
        out = self.out_proj(out)
        return out