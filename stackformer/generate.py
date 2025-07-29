import torch
import torch.nn.functional as F 

def text_generate(self, prompt_ids, max_context_len=128, max_new_tokens=50, temperature=1.0, top_k=None, top_p=1.0, eos_token_id=None):
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    generated = prompt_ids.clone()
    
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