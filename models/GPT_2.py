import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- position embedding ---
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        position = torch.arange(0, max_seq_len).unsqueeze(1)  # (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))  # (emb_dim/2)
        pe = torch.zeros(max_seq_len, emb_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [max_seq_len, emb_dim]
    def forward(self, x):
        B, T = x.shape
        return self.pe[:T].unsqueeze(0).expand(B, T, -1)


# --- Multi Head Attention ---
class MultiHeadAttention(nn.Module):  # Fixed typo
    def __init__(self, Emb_dim, num_heads, dropout, device='cpu', dtype=torch.float32):
        super().__init__()
        assert Emb_dim % num_heads == 0, "Emb_dim must be divisible by num_heads"
        self.Emb_dim = Emb_dim
        self.device = device
        self.num_heads = num_heads
        self.head_dim = Emb_dim // num_heads

        self.key = nn.Linear(Emb_dim, Emb_dim, bias=False, dtype=dtype, device=device)
        self.query = nn.Linear(Emb_dim, Emb_dim, bias=False, dtype=dtype, device=device)
        self.value = nn.Linear(Emb_dim, Emb_dim, bias=False, dtype=dtype, device=device)
        
        self.scale = torch.tensor(self.head_dim ** 0.5, device=device, dtype=dtype)
        self.out_proj = nn.Linear(Emb_dim, Emb_dim, dtype=dtype, device=device)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # Fixed parameter name
        Batch_size, Seq_len, _ = x.shape
        
        # Generate Q, K, V and reshape for multi-head attention
        Keys = self.key(x).view(Batch_size, Seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        Querys = self.query(x).view(Batch_size, Seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        Values = self.value(x).view(Batch_size, Seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = (Querys @ Keys.transpose(-2, -1)) / self.scale

        causal_mask = torch.triu(torch.ones(Seq_len, Seq_len, dtype=torch.bool, device=self.device), diagonal=1)
        scores = scores.masked_fill_(causal_mask[None, None, :, :], float('-inf'))

        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ Values
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(Batch_size, Seq_len, self.Emb_dim)
        
        return self.out_proj(out)

# --- Feed Forward ---
class FF_ReLU(nn.Module):
    def __init__(self, Emb_dim, hidden_dim, dropout=0.1, device='cpu', dtype=torch.float32):
        super().__init__()
        self.relu = nn.Sequential(
            nn.Linear(Emb_dim, hidden_dim, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(dropout),  # Added dropout
            nn.Linear(hidden_dim, Emb_dim, device=device, dtype=dtype),
        )
    
    def forward(self, x):
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

# --- single block --- FIXED TO PRE-NORM
class Block(nn.Module):  # Capitalized class name
    def __init__(self, Emb_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.attention = MultiHeadAttention(Emb_dim, num_heads, dropout, device=device, dtype=dtype)
        self.norm1 = LayerNorm(Emb_dim, eps=eps, device=device, dtype=dtype)
        self.ff_relu = FF_ReLU(Emb_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm2 = LayerNorm(Emb_dim, eps=eps, device=device, dtype=dtype)
        
    def forward(self, x):
        # Pre-norm: normalize before attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = x + residual  # Residual connection
        
        # Pre-norm: normalize before FF
        residual = x
        x = self.norm2(x)
        x = self.ff_relu(x)
        x = x + residual  # Residual connection
        
        return x

# --- Encoder ---
class Encoder(nn.Module):
    def __init__(self, num_layers, Emb_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.layers = nn.ModuleList([
            Block(Emb_dim, num_heads, dropout, hidden_dim, eps, device=device, dtype=dtype)
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
        
        # --- Embedding dropout ---
        self.emb_dropout = nn.Dropout(config.get('dropout', 0.1))
        
        # --- position embedding ---
        self.position_embedding = SinusoidalPositionalEmbedding(config['seq_len'], config['Emb_dim'])

        #  --- Encoder ---
        self.encoder = Encoder(
            num_layers=config['num_layers'],
            Emb_dim=config['Emb_dim'],
            num_heads=config['num_heads'],
            dropout=config.get('dropout', 0.1),
            hidden_dim=config['hidden_dim'],
            eps=config.get('eps', 1e-5),
            device=self.device,
            dtype=self.dtype
        )
        
        # --- Final norm        
        self.final_norm = LayerNorm(config['Emb_dim'], eps=config.get('eps', 1e-5),
                            device=self.device, dtype=self.dtype)
        
        # --- Output Projection ---
        self.lm_head = nn.Linear(config['Emb_dim'], vocab_size, bias=False, 
                    dtype=self.dtype, device=self.device)
    
    def forward(self, x):
        emb = self.embedding(x)
        pos = self.position_embedding(x)
        x = emb + pos
        x = self.emb_dropout(x)  # Added embedding dropout
        x = self.encoder(x)
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x
    
    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0, top_k=None, top_p=1.0):
        self.eval()

        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)  # (1, T)

        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            input_ids = generated[:, -self.config['seq_len']:]
            logits = self.forward(input_ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k is not None and top_k > 0:
                topk_vals, topk_indices = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float('-inf'))
                mask.scatter_(dim=-1, index=topk_indices, src=topk_vals)
                logits = mask

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(probs, dim=-1)

                sorted_mask = cum_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = 0

                indices_to_remove = sorted_mask.scatter(dim=-1, index=sorted_indices, src=sorted_mask)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            generated = torch.cat([generated, next_token], dim=-1)

        return generated