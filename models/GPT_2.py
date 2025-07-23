import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Adaptive position embedding ---
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, emb_dim, max_seq_len=10000):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        
        # Create initial positional encoding matrix
        self._create_pe_matrix(max_seq_len, emb_dim)
    
    def _create_pe_matrix(self, seq_len, emb_dim):
        """Create positional encoding matrix for given sequence length and embedding dimension"""
        position = torch.arange(0, seq_len).unsqueeze(1).float()  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * 
                           -(math.log(10000.0) / emb_dim))  # (emb_dim//2)
        
        pe = torch.zeros(seq_len, emb_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        if emb_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe)  # [seq_len, emb_dim]
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len) or (batch_size, seq_len, emb_dim)
        Returns:
            Positional embeddings of shape (batch_size, seq_len, emb_dim)
        """
        if x.dim() == 2:
            batch_size, seq_len = x.shape
        elif x.dim() == 3:
            batch_size, seq_len, _ = x.shape
        else:
            raise ValueError(f"Input tensor must be 2D or 3D, got {x.dim()}D")
        
        # Check if we need to extend the positional encoding
        if seq_len > self.pe.size(0):
            self._create_pe_matrix(seq_len, self.emb_dim)
            # Move to the same device as input
            self.pe = self.pe.to(x.device)
        
        # Return positional embeddings for the current sequence length
        pos_emb = self.pe[:seq_len].unsqueeze(0).expand(batch_size, seq_len, -1)
        return pos_emb.to(x.device)


# --- Multi Head Attention ---
class MultiHeadAttention(nn.Module):
    def __init__(self, Emb_dim, num_heads, dropout=0.1, device='cpu', dtype=torch.float32):
        super().__init__()
        assert Emb_dim % num_heads == 0, "Emb_dim must be divisible by num_heads"
        self.Emb_dim = Emb_dim
        self.device = device
        self.num_heads = num_heads
        self.head_dim = Emb_dim // num_heads

        self.key = nn.Linear(Emb_dim, Emb_dim, bias=False, dtype=dtype, device=device)
        self.query = nn.Linear(Emb_dim, Emb_dim, bias=False, dtype=dtype, device=device)
        self.value = nn.Linear(Emb_dim, Emb_dim, bias=False, dtype=dtype, device=device)
        
        self.scale = math.sqrt(self.head_dim)
        self.out_proj = nn.Linear(Emb_dim, Emb_dim, dtype=dtype, device=device)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Generate Q, K, V and reshape for multi-head attention
        keys = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = (queries @ keys.transpose(-2, -1)) / self.scale

        # Create causal mask dynamically based on current sequence length
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal_mask[None, None, :, :], float('-inf'))

        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ values
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.Emb_dim)
        
        return self.out_proj(out)

# --- Feed Forward ---
class FF_ReLU(nn.Module):
    def __init__(self, Emb_dim, hidden_dim, dropout=0.1, device='cpu', dtype=torch.float32):
        super().__init__()
        self.relu = nn.Sequential(
            nn.Linear(Emb_dim, hidden_dim, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(dropout),
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

# --- Transformer Block ---
class Block(nn.Module):
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
        
        if config.get('dtype') == 'float16':
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        # --- Token embedding ---
        self.embedding = nn.Embedding(vocab_size, config['Emb_dim'], dtype=self.dtype, device=self.device)
        
        # --- Embedding dropout ---
        self.emb_dropout = nn.Dropout(config.get('dropout', 0.1))
        
        # --- Adaptive position embedding ---
        self.position_embedding = SinusoidalPositionalEmbedding(
            config['Emb_dim'], 
            max_seq_len=config.get('max_seq_len', config.get('seq_len', 2048))
        )

        # --- Encoder ---
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
        # x shape: (batch_size, seq_len)
        emb = self.embedding(x)  # (batch_size, seq_len, emb_dim)
        pos = self.position_embedding(x)  # (batch_size, seq_len, emb_dim)
        x = emb + pos
        x = self.emb_dropout(x)
        x = self.encoder(x)
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x
    
    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0, top_k=None, top_p=1.0):
        self.eval()

        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)  # (1, seq_len)

        generated = prompt_ids.clone()
        max_context_len = self.config.get('max_seq_len', self.config.get('seq_len', 2048))

        for _ in range(max_new_tokens):
            # Use sliding window if sequence gets too long
            if generated.size(1) > max_context_len:
                input_ids = generated[:, -max_context_len:]
            else:
                input_ids = generated
                
            logits = self.forward(input_ids)  # (batch_size, seq_len, vocab_size)
            logits = logits[:, -1, :]  # (batch_size, vocab_size)

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
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            generated = torch.cat([generated, next_token], dim=-1)

        return generated