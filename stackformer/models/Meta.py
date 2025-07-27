import torch
from torch import nn
import torch.nn.functional as F

from stackformer.modules.Attention import kv_cache_group_query
from stackformer.modules.Feed_forward import FF_SiLU
from stackformer.modules.Normalization import RMSNormilization

class block(nn.Module):
    def __init__(self, emb_dim, query_num_heads, kv_num_heads, batch_size, kv_seq_len, hidden_dim,
                eps=1e-5, dropout=0.1, dtype=torch.float32, device='cpu'):
        super().__init__()
        self.attn_norm = RMSNormilization(dim=emb_dim, eps=eps)
        self.ff_norm = RMSNormilization(dim=emb_dim, eps=eps)
        self.attn = kv_cache_group_query(emb_dim=emb_dim, query_num_heads=query_num_heads, kv_num_heads=kv_num_heads,
                                        batch_size=batch_size, kv_seq_len=kv_seq_len, dtype=dtype,
                                        dropout=dropout, device=device)
        self.ff = FF_SiLU(emb_dim=emb_dim, hidden_dim=hidden_dim, device=device, dtype=dtype)

    def forward(self, x, start_pos):
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)
        x = x + residual

        residual = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residual
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, emb_dim, query_num_heads, kv_num_heads, batch_size, kv_seq_len,
                hidden_dim, eps=1e-5, dropout=0.1, dtype=torch.float32, device='cpu'):
        super().__init__()
        self.layers = nn.ModuleList([
            block(emb_dim=emb_dim, query_num_heads=query_num_heads, kv_num_heads=kv_num_heads,
                batch_size=batch_size, kv_seq_len=kv_seq_len, hidden_dim=hidden_dim,
                eps=eps, dropout=dropout, dtype=dtype, device=device)
            for _ in range(num_layers)
        ])

    def forward(self, x, start_pos):
        for layer in self.layers:
            x = layer(x, start_pos)
        return x


class Llama_2(nn.Module):
    def __init__(self, num_layers, emb_dim, query_num_heads, kv_num_heads, batch_size, kv_seq_len, vocab_size,
                hidden_dim, eps=1e-5, dropout=0.1, dtype=torch.float32, device='cpu'):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.seq_len = kv_seq_len  # For generation slicing

        self.embedding = nn.Embedding(vocab_size, emb_dim, dtype=dtype, device=device)

        self.encoder = Encoder(num_layers=num_layers, emb_dim=emb_dim, query_num_heads=query_num_heads,
                            kv_num_heads=kv_num_heads, batch_size=batch_size, kv_seq_len=kv_seq_len,
                            hidden_dim=hidden_dim, eps=eps, dropout=dropout, dtype=dtype, device=device)

        self.final_norm = RMSNormilization(emb_dim, eps=eps)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False, dtype=dtype, device=device)

    def forward(self, input_ids, start_pos=0):
        x = self.embedding(input_ids)
        x = self.encoder(x, start_pos)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0, top_k=None, top_p=1.0):
        self.eval()
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        generated = prompt_ids.clone()
        for step in range(max_new_tokens):
            input_ids = generated[:, -self.seq_len:]
            logits = self.forward(input_ids, start_pos=step)  # Correct start_pos
            logits = logits[:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            if top_k is not None and top_k > 0:
                topk_vals, topk_indices = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float('-inf'))
                mask.scatter_(dim=-1, index=topk_indices, src=topk_vals)
                logits = mask

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(probs, dim=-1)
                sorted_mask = cum_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = 0
                indices_to_remove = sorted_mask.scatter(dim=-1, index=sorted_indices, src=sorted_mask)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

        return generated