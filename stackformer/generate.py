from __future__ import annotations

import torch
import torch.nn.functional as F


def text_generate(
    self,
    prompt_ids: torch.Tensor,
    max_context_len: int = 128,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    generated = prompt_ids.clone()

    for _ in range(max_new_tokens):
        if generated.size(1) > max_context_len:
            input_ids = generated[:, -max_context_len:]
        else:
            input_ids = generated

        logits = self.forward(input_ids)
        logits = logits[:, -1, :]

        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None and top_k > 0:
            topk_vals, topk_indices = torch.topk(logits, top_k)
            mask = torch.full_like(logits, float("-inf"))
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
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=-1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return generated
