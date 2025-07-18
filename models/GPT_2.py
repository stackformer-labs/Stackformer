import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- position embedding ---
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, seq_len, Emb_dim):
        super().__init__()
        self.seq_len = seq_len
        self.Emb_dim = Emb_dim

        position = torch.arange(0, seq_len).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, Emb_dim, 2) * -(math.log(10000.0) / Emb_dim))  # (Emb_dim/2)

        pe = torch.zeros(seq_len, Emb_dim)  # (seq_len, Emb_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it's moved with model
        self.register_buffer("pe", pe)

    def forward(self, batch_size=None):
        # Return with shape (B, T, D) if batch_size is given
        if batch_size is not None:
            return self.pe.unsqueeze(0).expand(batch_size, -1, -1)  # (B, T, D)
        else:
            return self.pe  # (T, D)

# --- Multi Head Attedtion ---
class Multi_Head_Attention(nn.Module):
    def __init__(self, Emb_dim, num_heads, dropout, device='cpu',dtype=torch.float32):
        super().__init__()
        assert Emb_dim % num_heads == 0, "Emb_dim must be divisible by num_heads"
        self.Emb_dim = Emb_dim
        self.device = device
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
        causal_mask = torch.triu(torch.ones(Seq_len, Seq_len, dtype=torch.bool, device=self.device), diagonal=1)
        scores = scores.masked_fill_(causal_mask[None, None, :, :], float('-inf'))

        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ Values  # (Batch_size, nh, Seq_len, hd)
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(Batch_size, Seq_len, self.Emb_dim)  # (Batch_size, Seq_len, Emb_dim)
        
        return self.out_proj(out)

# --- Feed Forward ---
class FF_ReLU(nn.Module):
    def __init__(self,Emb_dim,hidden_dim,device='cpu',dtype=torch.float32):
        super().__init__()
        self.relu=nn.Sequential(
            nn.Linear(Emb_dim,hidden_dim,device=device,dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim,Emb_dim,device=device,dtype=dtype),
        )
    def forward(self,x):
        return self.relu(x)

class LayerNorm(nn.Module):
    def __init__(self, Emb_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(Emb_dim, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(Emb_dim, device=device, dtype=dtype))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return norm_x * self.weight + self.bias


# --- single block ---
class block(nn.Module):
    def __init__(self, Emb_dim, num_heads, dropout, hidden_dim, esp=1e-5, device='cpu',dtype=torch.float32):
        super().__init__()
        self.attentation = Multi_Head_Attention(Emb_dim, num_heads, dropout, device=device,dtype=dtype)
        self.norm1 = LayerNorm(Emb_dim, eps=esp, device=device, dtype=dtype)
        self.ff_relu = FF_ReLU(Emb_dim, hidden_dim,device=device,dtype=dtype)
        self.norm2 = LayerNorm(Emb_dim, eps=esp, device=device, dtype=dtype)
        
    def forward(self, x):
        residual = x
        x = self.attentation(x)
        x = self.norm1(x + residual)
        
        residual = x
        x = self.ff_relu(x)
        x = self.norm2(x + residual)
        return x

# --- Encoder ---
class Encoder(nn.Module):
    def __init__(self, num_layers, Emb_dim, num_heads, dropout, hidden_dim, esp=1e-5, device='cpu',dtype=torch.float32):
        super().__init__()
        self.layers = nn.ModuleList([
            block(Emb_dim, num_heads, dropout, hidden_dim, esp, device=device,dtype=dtype)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class GPTModel(nn.Module):
    def __init__(self, config, vocab_size, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        self.vocab_size = vocab_size
        
        if config['dtype'] == 'float16':
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        # --- Token embedding ---
        self.embedding = nn.Embedding(vocab_size, config['Emb_dim'], dtype=self.dtype, device=self.device)
        
        # --- position embedding ---
        self.position_embedding = SinusoidalPositionalEmbedding(config['seq_len'], config['Emb_dim'])

        #  --- Encoder ---
        self.encoder = Encoder(
            num_layers=config['num_layers'],
            Emb_dim=config['Emb_dim'],
            num_heads=config['num_heads'],
            dropout=config.get('dropout', 0.1),
            hidden_dim=config['hidden_dim'],
            esp=config.get('eps', 1e-5),
            device=self.device,
            dtype=self.dtype
        )
        
        # --- Final norm        
        self.final_norm = LayerNorm(config['Emb_dim'], eps=config.get('eps', 1e-5),
                            device=self.device, dtype=self.dtype)

        
        # --- Output Projection ---
        self.lm_head = nn.Linear(config['Emb_dim'], vocab_size, bias=False, 
                    dtype=self.dtype, device=self.device)
    
    def forward(self, x, batch_size=None):
        emb = self.embedding(x)
        pos = self.position_embedding(batch_size=x.size(0))
        x = emb + pos
        x = self.encoder(x)
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x