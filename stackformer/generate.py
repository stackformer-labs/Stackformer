from __future__ import annotations

import torch
import torch.nn.functional as F

from stackformer.config import GenerationConfig


def _sample_next_token(logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    if temperature != 1.0:
        logits = logits / temperature

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        topk_vals, topk_indices = torch.topk(logits, top_k)
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(dim=-1, index=topk_indices, src=topk_vals)
        logits = filtered

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)

        sorted_mask = cum_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        remove_mask = torch.zeros_like(sorted_mask)
        remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
        logits = logits.masked_fill(remove_mask, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _resolve_generation_config(
    max_context_len: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float,
    eos_token_id: int | None,
    generation_config: GenerationConfig | None,
) -> GenerationConfig:
    if generation_config is None:
        return GenerationConfig(
            max_context_len=max_context_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )
    return generation_config


def text_generate(
    self,
    prompt_ids: torch.Tensor,
    max_context_len: int = 128,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    generation_config: GenerationConfig | None = None,
) -> torch.Tensor:
    config = _resolve_generation_config(
        max_context_len=max_context_len,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_token_id,
        generation_config=generation_config,
    )

    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    generated = prompt_ids.clone()
    batch_size = generated.size(0)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)

    # Cache-aware path when model provides prefill/decode APIs.
    cache = None
    use_cache = hasattr(self, "prefill") and hasattr(self, "decode")

    if use_cache:
        input_ids = generated[:, -config.max_context_len :]
        prefill_logits, cache = self.prefill(input_ids)
        logits = prefill_logits[:, -1, :]
    else:
        input_ids = generated[:, -config.max_context_len :]
        logits = self.forward(input_ids)[:, -1, :]

    for _ in range(config.max_new_tokens):
        next_token = _sample_next_token(logits, config.temperature, config.top_k, config.top_p)

        if config.eos_token_id is not None:
            eos_hits = next_token.squeeze(-1) == config.eos_token_id
            finished |= eos_hits

        generated = torch.cat([generated, next_token], dim=-1)

        if finished.all():
            break

        if use_cache:
            decode_logits, cache = self.decode(next_token, cache)
            logits = decode_logits[:, -1, :]
        else:
            input_ids = generated[:, -config.max_context_len :]
            logits = self.forward(input_ids)[:, -1, :]

    return generated
