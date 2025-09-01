import torch
import torch.nn as nn
import torch.nn.functional as F

from stackformer.modules.Attention import Multi_Head_Attention
from stackformer.modules.position_embedding import AbsolutePositionEmbedding
from stackformer.modules.Normalization import LayerNormalization
from stackformer.modules.Feed_forward import FF_GELU
from stackformer.generate import text_generate

'''
GPT-1
Attention: MHA
Mask: Casual
position: absolute
FF: GeLU
Norm: post normalization (layer norm)
'''
# --- GPT_1 Encoder Block ---
class GPT_1_Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.attention = Multi_Head_Attention(embed_dim, num_heads, dropout, device=device, dtype=dtype)
        self.norm1 = LayerNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        self.FF_GELU = FF_GELU(embed_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm2 = LayerNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        
    def forward(self, x):
        residual = x
        x = self.attention(x)
        x = self.norm1(x)
        x = x + residual
        
        residual = x
        x = self.FF_GELU(x)
        x = self.norm2(x)
        x = x + residual
        
        return x

# --- GPT_1 Encoder ---
class GPT_1_Encoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.layers = nn.ModuleList([
            GPT_1_Block(embed_dim, num_heads, dropout, hidden_dim, eps, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class GPT_1(nn.Module):
    def __init__(self, vocab_size, num_layers, embed_dim, num_heads, seq_len,
            dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.seq_len = seq_len
        
        # --- Token embedding ---
        self.embedding = nn.Embedding(vocab_size, embed_dim, dtype=self.dtype, device=self.device)
        
        # --- absolute position embedding ---
        self.position_embedding = AbsolutePositionEmbedding(embed_dim=embed_dim, seq_len=seq_len, device=device,dtype=dtype)
        
        # --- Encoder ---
        self.encoder = GPT_1_Encoder(num_layers=num_layers,embed_dim=embed_dim,num_heads=num_heads,dropout=dropout,
            hidden_dim=hidden_dim,eps=eps,device=self.device,dtype=self.dtype)
        
        # --- Final norm        
        self.final_norm = LayerNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        
        # --- Output Projection ---
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, dtype=self.dtype, device=self.device)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        emb = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        pos = self.position_embedding(x)  # (batch_size, seq_len, embed_dim)
        x = emb + pos
        x = self.encoder(x)
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x
        
    @torch.no_grad()
    def generate(self, prompt_ids, max_context_len=128, max_new_tokens=50, temperature=1.0, top_k=None, top_p=1.0, eos_token_id=None):
        return text_generate(self, prompt_ids, max_context_len, max_new_tokens, temperature, top_k, top_p, eos_token_id)
    
'''
GPT-2
Attention: MHA
Mask: Casual
position: absolute
FF: GeLU
Norm: pre normalization (layer norm)
'''
# --- Encoder Block ---
class GPT_2_Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.attention = Multi_Head_Attention(embed_dim, num_heads, dropout, device=device, dtype=dtype)
        self.norm1 = LayerNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        self.FF_GELU = FF_GELU(embed_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm2 = LayerNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.FF_GELU(x)
        x = x + residual
        
        return x

# --- Encoder ---
class GPT_2_Encoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.layers = nn.ModuleList([
            GPT_2_Block(embed_dim, num_heads, dropout, hidden_dim, eps, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class GPT_2(nn.Module):
    def __init__(self, vocab_size, num_layers, embed_dim, num_heads, seq_len,
            dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.seq_len = seq_len
        
        # --- Token embedding ---
        self.embedding = nn.Embedding(vocab_size, embed_dim, dtype=self.dtype, device=self.device)
        
        # --- Adaptive position embedding ---
        self.position_embedding = AbsolutePositionEmbedding(embed_dim=embed_dim, seq_len=seq_len, device=device,dtype=dtype)
        
        # --- Encoder ---
        self.encoder = GPT_2_Encoder(num_layers=num_layers,embed_dim=embed_dim,num_heads=num_heads,dropout=dropout,
            hidden_dim=hidden_dim,eps=eps,device=self.device,dtype=self.dtype)
        
        # --- Final norm        
        self.final_norm = LayerNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        
        # --- Output Projection ---
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, 
                    dtype=self.dtype, device=self.device)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        emb = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        pos = self.position_embedding(x)  # (batch_size, seq_len, embed_dim)
        x = emb + pos
        x = self.encoder(x)
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x
        
    @torch.no_grad()
    def generate(self, prompt_ids, max_context_len=128, max_new_tokens=50, temperature=1.0, top_k=None, top_p=1.0, eos_token_id=None):
        return text_generate(self, prompt_ids, max_context_len, max_new_tokens, temperature, top_k, top_p, eos_token_id)