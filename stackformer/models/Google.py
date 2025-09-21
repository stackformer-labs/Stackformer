import torch
from torch import nn
import torch.nn.functional as F

from stackformer.modules.Attention import Multi_query_Attention_With_RoPE,Multi_Head_Attention_With_RoPE
from stackformer.modules.Feed_forward import FF_GeGLU
from stackformer.modules.Normalization import RMSNormalization
from stackformer.generate import text_generate

'''
Gemma 1 2B:
Attention: MQH
Mask: Casual
position: RoPE
FF: GeGLU
Norm: pre norm (RMS norm)
'''
class gemma_1_2b_block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.attention = Multi_query_Attention_With_RoPE(embed_dim, num_heads, dropout, device=device, dtype=dtype)
        self.norm1 = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        self.FF_SwiGLU = FF_GeGLU(embed_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm2 = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.FF_SwiGLU(x)
        x = x + residual
        
        return x

# --- Encoder ---
class gemma_1_2b_Encoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.layers = nn.ModuleList([
            gemma_1_2b_block(embed_dim, num_heads, dropout, hidden_dim, eps, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class gemma_1_2b(nn.Module):
    def __init__(self, vocab_size, num_layers, embed_dim, num_heads, seq_len,
            dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.seq_len = seq_len
        
        # --- Token embedding ---
        self.embedding = nn.Embedding(vocab_size, embed_dim, dtype=self.dtype, device=self.device)
        
        # --- Encoder ---
        self.encoder = gemma_1_2b_Encoder(num_layers=num_layers,embed_dim=embed_dim,num_heads=num_heads,dropout=dropout,
                                    hidden_dim=hidden_dim,eps=eps,device=self.device,dtype=self.dtype)
        
        # --- Final norm        
        self.final_norm = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        
        # --- Output Projection ---
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, dtype=self.dtype, device=self.device)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        emb = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = self.encoder(emb)
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x
        
    @torch.no_grad()
    def generate(self, prompt_ids, max_context_len=128, max_new_tokens=50, temperature=1.0, top_k=None, top_p=1.0, eos_token_id=None):
        return text_generate(self, prompt_ids, max_context_len, max_new_tokens, temperature, top_k, top_p, eos_token_id)

'''
Gemma 1 7B:
Attention: MHA
Mask: Casual
position: RoPE
FF: GeGLU
Norm: pre norm (RMS norm)
'''
class gemma_1_7b_block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.attention = Multi_Head_Attention_With_RoPE(embed_dim, num_heads, dropout, device=device, dtype=dtype)
        self.norm1 = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        self.FF_SwiGLU = FF_GeGLU(embed_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm2 = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.FF_SwiGLU(x)
        x = x + residual
        
        return x
    
# --- Encoder ---
class gemma_1_7b_Encoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.layers = nn.ModuleList([
            gemma_1_7b_block(embed_dim, num_heads, dropout, hidden_dim, eps, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class gemma_1_7b(nn.Module):
    def __init__(self, vocab_size, num_layers, embed_dim, num_heads, seq_len,
            dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.seq_len = seq_len
        
        # --- Token embedding ---
        self.embedding = nn.Embedding(vocab_size, embed_dim, dtype=self.dtype, device=self.device)
        
        # --- Encoder ---
        self.encoder = gemma_1_7b_Encoder(num_layers=num_layers,embed_dim=embed_dim,num_heads=num_heads,dropout=dropout,
                                    hidden_dim=hidden_dim,eps=eps,device=self.device,dtype=self.dtype)
        
        # --- Final norm        
        self.final_norm = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        
        # --- Output Projection ---
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, dtype=self.dtype, device=self.device)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        emb = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = self.encoder(emb)
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x
        
    @torch.no_grad()
    def generate(self, prompt_ids, max_context_len=128, max_new_tokens=50, temperature=1.0, top_k=None, top_p=1.0, eos_token_id=None):
        return text_generate(self, prompt_ids, max_context_len, max_new_tokens, temperature, top_k, top_p, eos_token_id)