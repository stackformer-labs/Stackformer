import math
import torch
import torch.nn as nn
import torch.nn.functional as F 

from stackformer.modules.Attention import Multi_Head_Attention, Cross_MultiHead_Attention
from stackformer.modules.position_embedding import SinusoidalPositionalEmbedding
from stackformer.modules.Feed_forward import FF_ReLU
from stackformer.modules.Normalization import LayerNorm

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