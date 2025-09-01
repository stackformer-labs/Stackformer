import torch
from torch import nn
import torch.nn.functional as F

from stackformer.modules.Attention import Multi_Head_Attention,kv_cache_group_query
from stackformer.modules.Feed_forward import FF_SwiGLU
from stackformer.modules.Normalization import RMSNormalization
from stackformer.generate import text_generate

'''
llama 1
Attention: MHA
Mask: Casual
position: RoPE
FF: SwiGLU
Norm: pre norm (RMS norm)
'''
class llama_1_Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.attention = Multi_Head_Attention(embed_dim, num_heads, dropout, device=device, dtype=dtype)
        self.norm1 = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        self.FF_SwiGLU = FF_SwiGLU(embed_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm2 = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attention(x,rope=True)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.FF_SwiGLU(x)
        x = x + residual
        
        return x

# --- Encoder ---
class llama_1_Encoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.layers = nn.ModuleList([
            llama_1_Block(embed_dim, num_heads, dropout, hidden_dim, eps, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class llama_1(nn.Module):
    def __init__(self, vocab_size, num_layers, embed_dim, num_heads, seq_len,
            dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.seq_len = seq_len
        
        # --- Token embedding ---
        self.embedding = nn.Embedding(vocab_size, embed_dim, dtype=self.dtype, device=self.device)
        
        # --- Encoder ---
        self.encoder = llama_1_Encoder(num_layers=num_layers,embed_dim=embed_dim,num_heads=num_heads,dropout=dropout,
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
llama 2
Attention: GQA with KV catch
Mask: Casual
position: RoPE
FF: SwiGLU
Norm: pre norm (RMS norm)
'''
class llama_2_Block(nn.Module):
    def __init__(self, embed_dim, num_query_heads, num_kv_heads, batch_size, kv_seq_len, hidden_dim,
                eps=1e-5, dropout=0.1, dtype=torch.float32, device='cpu'):
        super().__init__()
        self.attn_norm = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        self.attn = kv_cache_group_query(embed_dim=embed_dim, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads,
                                        batch_size=batch_size, kv_seq_len=kv_seq_len, dtype=dtype, dropout=dropout, device=device)
        self.ff_norm = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        self.ff = FF_SwiGLU(embed_dim=embed_dim, hidden_dim=hidden_dim, device=device, dtype=dtype)

    def forward(self, x, start_pos):
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, start_pos, rope=True)
        x = x + residual

        residual = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residual
        return x

class llama_2_Encoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_query_heads, num_kv_heads, batch_size, kv_seq_len,
                hidden_dim, eps=1e-5, dropout=0.1, dtype=torch.float32, device='cpu'):
        super().__init__()
        self.layers = nn.ModuleList([
            llama_2_Block(embed_dim=embed_dim, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads,
                batch_size=batch_size, kv_seq_len=kv_seq_len, hidden_dim=hidden_dim,
                eps=eps, dropout=dropout, dtype=dtype, device=device)
            for _ in range(num_layers)
        ])

    def forward(self, x, start_pos):
        for layer in self.layers:
            x = layer(x, start_pos)
        return x

class llama_2(nn.Module):
    def __init__(self, num_layers, embed_dim, num_query_heads, num_kv_heads, batch_size, kv_seq_len, vocab_size,
                hidden_dim, eps=1e-5, dropout=0.1, dtype=torch.float32, device='cpu'):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.seq_len = kv_seq_len  # For generation slicing

        self.embedding = nn.Embedding(vocab_size, embed_dim, dtype=dtype, device=device)

        self.llama_2_Encoder = llama_2_Encoder(num_layers=num_layers, embed_dim=embed_dim, num_query_heads=num_query_heads,
                            num_kv_heads=num_kv_heads, batch_size=batch_size, kv_seq_len=kv_seq_len,
                            hidden_dim=hidden_dim, eps=eps, dropout=dropout, dtype=dtype, device=device)

        self.final_norm = RMSNormalization(embed_dim, eps=eps, device=device,dtype=dtype)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, dtype=dtype, device=device)

    def forward(self, input_ids, start_pos=0):
        x = self.embedding(input_ids)
        x = self.llama_2_Encoder(x, start_pos)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_context_len=128, max_new_tokens=50, temperature=1.0, top_k=None, top_p=1.0, eos_token_id=None):
        return text_generate(self, prompt_ids, max_context_len, max_new_tokens, temperature, top_k, top_p, eos_token_id)