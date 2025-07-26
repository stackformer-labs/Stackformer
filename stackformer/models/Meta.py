import torch
from torch import nn
import torch.nn.functional as F


def precompute_theta_position_frequency(head_dim, seq_len, device='cpu', theta=10000.0):
    assert head_dim % 2 == 0, "head_dim must be even"
    theta_numerator = torch.arange(0, head_dim, 2, device=device)
    inv_freq = 1.0 / (theta ** (theta_numerator / head_dim))
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, inv_freq)
    freq_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freq_complex


def apply_rotry_position_embedding(x, freq_complex, device='cpu', dtype=torch.float32):
    batch_size, seq_len, num_head, emb_dim = x.shape
    assert emb_dim % 2 == 0, "emb_dim must be even"
    x_reshaped = x.view(batch_size, seq_len, num_head, emb_dim // 2, 2).to(device=device, dtype=dtype)
    x_complex = torch.view_as_complex(x_reshaped)
    freq_complex = freq_complex[:seq_len].unsqueeze(0).unsqueeze(2).to(device=device)
    x_rotated = x_complex * freq_complex
    x_out = torch.view_as_real(x_rotated).contiguous().view(batch_size, seq_len, num_head, emb_dim)
    return x_out.to(device=device, dtype=dtype)


class kv_cache_group_query(nn.Module):
    def __init__(self, emb_dim, query_num_heads, kv_num_heads, batch_size, kv_seq_len,
                device='cpu', dtype=torch.float32, dropout=0.1):
        super().__init__()
        assert emb_dim % query_num_heads == 0, "Embedding dim must be divisible by query heads"
        assert query_num_heads % kv_num_heads == 0, "query heads must be divisible by kv heads"

        self.device = device
        self.dtype = dtype
        self.emb_dim = emb_dim
        self.query_num_heads = query_num_heads
        self.kv_num_heads = kv_num_heads
        self.head_dim = emb_dim // query_num_heads
        self.num_queries_per_kv = query_num_heads // kv_num_heads
        self.kv_seq_len = kv_seq_len

        self.query = nn.Linear(emb_dim, emb_dim, bias=False, dtype=dtype, device=device)
        self.key = nn.Linear(emb_dim, kv_num_heads * self.head_dim, bias=False, dtype=dtype, device=device)
        self.value = nn.Linear(emb_dim, kv_num_heads * self.head_dim, bias=False, dtype=dtype, device=device)

        self.out_proj = nn.Linear(query_num_heads * self.head_dim, emb_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("cache_keys", torch.zeros(batch_size, kv_seq_len, kv_num_heads, self.head_dim, device=device, dtype=dtype))
        self.register_buffer("cache_value", torch.zeros(batch_size, kv_seq_len, kv_num_heads, self.head_dim, device=device, dtype=dtype))

    def forward(self, x, start_pos):
        batch_size, seq_len, _ = x.shape

        xq = self.query(x).view(batch_size, seq_len, self.query_num_heads, self.head_dim)
        xk = self.key(x).view(batch_size, seq_len, self.kv_num_heads, self.head_dim)
        xv = self.value(x).view(batch_size, seq_len, self.kv_num_heads, self.head_dim)

        freq_q = precompute_theta_position_frequency(head_dim=self.head_dim, seq_len=seq_len, device=self.device)
        xq = apply_rotry_position_embedding(xq, freq_q, device=self.device, dtype=self.dtype)
        freq_k = precompute_theta_position_frequency(head_dim=self.head_dim, seq_len=self.kv_seq_len, device=self.device)
        xk = apply_rotry_position_embedding(xk, freq_k, device=self.device, dtype=self.dtype)

        self.cache_keys[:, start_pos:start_pos + seq_len] = xk
        self.cache_value[:, start_pos:start_pos + seq_len] = xv

        xk_full = self.cache_keys[:, :start_pos + seq_len]
        xv_full = self.cache_value[:, :start_pos + seq_len]

        query = xq.transpose(1, 2)
        key = xk_full.transpose(1, 2).repeat_interleave(self.num_queries_per_kv, dim=1)
        value = xv_full.transpose(1, 2).repeat_interleave(self.num_queries_per_kv, dim=1)

        attn_scores = torch.matmul(query, key.transpose(2, 3)) / (self.head_dim ** 0.5)

        causal_mask = torch.triu(torch.ones(seq_len, attn_scores.shape[-1], dtype=torch.bool, device=self.device), diagonal=1)
        attn_scores.masked_fill_(causal_mask[None, None, :, :], float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, value)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.query_num_heads * self.head_dim)
        return self.dropout(self.out_proj(out))


class RMSNormilization(nn.Module):
    def __init__(self, emb_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim, dtype=dtype, device=device))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm / (x.shape[-1] ** 0.5)
        return (x / (rms + self.eps)) * self.scale


class FF_SiLU(nn.Module):
    def __init__(self, emb_dim, hidden_dim, device='cpu', dtype=torch.float32):
        super().__init__()
        self.silu = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_dim, emb_dim, device=device, dtype=dtype),
        )

    def forward(self, x):
        return self.silu(x)


class block(nn.Module):
    def __init__(self, emb_dim, query_num_heads, kv_num_heads, batch_size, kv_seq_len, hidden_dim,
                 eps=1e-5, dropout=0.1, dtype=torch.float32, device='cpu'):
        super().__init__()
        self.attn_norm = RMSNormilization(emb_dim=emb_dim, eps=eps, device=device, dtype=dtype)
        self.ff_norm = RMSNormilization(emb_dim=emb_dim, eps=eps, device=device, dtype=dtype)
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

        self.final_norm = RMSNormilization(emb_dim, eps=eps, device=device, dtype=dtype)
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