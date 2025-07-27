import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from stackformer.modules.Attention import Multi_Head_Attention
from stackformer.modules.position_embedding import SinusoidalPositionalEmbedding
from stackformer.modules.Normalization import LayerNorm
from stackformer.modules.Feed_forward import FF_ReLU

# --- Encoder Block ---
class Block(nn.Module):
    def __init__(self, Emb_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.attention = Multi_Head_Attention(Emb_dim, num_heads, dropout, device=device, dtype=dtype)
        self.norm1 = LayerNorm(Emb_dim, eps=eps, device=device, dtype=dtype)
        self.ff_relu = FF_ReLU(Emb_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm2 = LayerNorm(Emb_dim, eps=eps, device=device, dtype=dtype)
        
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

class GPT_2(nn.Module):
    def __init__(self, vocab_size, num_layers, Emb_dim, num_heads, seq_len,
            dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.seq_len = seq_len
        
        # --- Token embedding ---
        self.embedding = nn.Embedding(vocab_size, Emb_dim, dtype=self.dtype, device=self.device)
        
        # --- Embedding dropout ---
        self.emb_dropout = nn.Dropout(dropout)
        
        # --- Adaptive position embedding ---
        self.position_embedding = SinusoidalPositionalEmbedding(
            emb_dim=Emb_dim, 
            seq_len=seq_len
        )

        # --- Encoder ---
        self.encoder = Encoder(
            num_layers=num_layers,
            Emb_dim=Emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            hidden_dim=hidden_dim,
            eps=eps,
            device=self.device,
            dtype=self.dtype
        )
        
        # --- Final norm        
        self.final_norm = LayerNorm(Emb_dim, eps=eps,
                            device=self.device, dtype=self.dtype)
        
        # --- Output Projection ---
        self.lm_head = nn.Linear(Emb_dim, vocab_size, bias=False, 
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
    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0, top_k=None, top_p=1.0, eos_token_id=None):
        self.eval()
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)  # (1, seq_len)
            
        generated = prompt_ids.clone()
        max_context_len = self.seq_len

        for _ in range(max_new_tokens):
            # Use sliding window if sequence gets too long
            if generated.size(1) > max_context_len:
                input_ids = generated[:, -max_context_len:]
            else:
                input_ids = generated
                
            logits = self.forward(input_ids)  # (batch_size, seq_len, vocab_size)
            logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # --- Temperature scaling ---
            if temperature != 1.0:
                logits = logits / temperature

            # --- Top-k filtering ---
            if top_k is not None and top_k > 0:
                topk_vals, topk_indices = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float('-inf'))
                mask.scatter_(dim=-1, index=topk_indices, src=topk_vals)
                logits = mask

            # --- Top-p ---
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

            # check if we've reached the end of the sequence
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return generated