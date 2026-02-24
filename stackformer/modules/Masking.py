"""
masks.py
--------

Attention mask utilities for dense and sparse transformer layouts.

All masks follow PyTorch SDPA convention:

    True  -> masked (blocked)
    False -> visible (allowed)

Returned shape:
    (seq_len, seq_len)

This module supports composing multiple masking strategies using
a registry-based factory pattern.
"""

from typing import Callable, Dict, List
import torch


# --- Base Mask Functions ---
def causal(seq_len: int) -> torch.Tensor:
    """
    Standard autoregressive causal mask.

    Prevents attending to future tokens.

    Args:
        seq_len (int): Sequence length.

    Returns:
        torch.Tensor: (seq_len, seq_len) boolean mask.
    """
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool),
        diagonal=1
    )


def sliding_window(seq_len: int, window_size: int) -> torch.Tensor:
    """
    Sliding window causal attention.

    Each token attends only to the previous `window_size` tokens.

    Args:
        seq_len (int)
        window_size (int)

    Returns:
        torch.Tensor
    """
    causal_lower = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool)
    )
    band = torch.triu(
        causal_lower,
        diagonal=-(window_size - 1)
    )
    return ~band


def dilated_causal(seq_len: int, dilation: int) -> torch.Tensor:
    """
    Dilated causal attention.

    Tokens attend to previous tokens spaced by `dilation`.

    Args:
        seq_len (int)
        dilation (int)

    Returns:
        torch.Tensor
    """
    i = torch.arange(seq_len).unsqueeze(1)
    j = torch.arange(seq_len).unsqueeze(0)
    visible = (i >= j) & ((i - j) % dilation == 0)
    return ~visible


def random_mask(seq_len: int, num_random: int) -> torch.Tensor:
    """
    Random sparse causal attention.

    Each token attends to `num_random` random past tokens.

    Args:
        seq_len (int)
        num_random (int)

    Returns:
        torch.Tensor
    """
    visible = torch.zeros(seq_len, seq_len, dtype=torch.bool)

    for i in range(seq_len):
        candidates = list(range(i))
        if not candidates:
            continue

        idx = torch.randperm(len(candidates))[:num_random]
        selected = torch.tensor(candidates)[idx]
        visible[i, selected] = True

    return ~visible


def global_mask(seq_len: int, global_index: List[int]) -> torch.Tensor:
    """
    Global attention mask.

    Selected indices attend to all tokens and are attended by all.

    Args:
        seq_len (int)
        global_index (List[int])

    Returns:
        torch.Tensor
    """
    visible = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    g = torch.tensor(global_index)

    visible[g, :] = True
    visible[:, g] = True

    return ~visible


def full_mask(seq_len: int) -> torch.Tensor:
    """
    Full attention (no masking).

    Args:
        seq_len (int)

    Returns:
        torch.Tensor
    """
    return torch.zeros(seq_len, seq_len, dtype=torch.bool)


def mistral(seq_len: int, window_size: int, dilation: int) -> torch.Tensor:
    """
    Mistral-style hybrid mask:
        sliding window + dilated attention

    Args:
        seq_len (int)
        window_size (int)
        dilation (int)

    Returns:
        torch.Tensor
    """
    return sliding_window(seq_len, window_size) | \
           dilated_causal(seq_len, dilation)


# --- Registry Pattern ---

MASK_REGISTRY: Dict[str, Callable] = {
    "causal": causal,
    "sliding_window": sliding_window,
    "full": full_mask,
    "dilated_causal": dilated_causal,
    "random_mask": random_mask,
    "global_mask": global_mask,
    "mistral": mistral,
}


# --- Unified Factory ---

def make_mask(
    mask_types: List[str],
    seq_len: int,
    device: torch.device | None = None,
    **kwargs
) -> torch.Tensor:
    """
    Unified mask factory.

    Combines multiple mask patterns into a single boolean mask.

    Args:
        mask_types (List[str]):
            List of mask names to combine.

        seq_len (int):
            Sequence length.

        device (torch.device, optional):
            Device for mask tensor.

        **kwargs:
            Additional parameters passed automatically:
                window_size (int)
                dilation (int)
                num_random (int)
                global_index (List[int])

    Returns:
        torch.Tensor:
            Boolean mask (seq_len, seq_len)
    """

    if not isinstance(mask_types, (list, tuple)):
        raise TypeError("mask_types must be a list of strings")

    mask = torch.zeros(
        seq_len,
        seq_len,
        dtype=torch.bool,
        device=device
    )

    for name in mask_types:
        if name not in MASK_REGISTRY:
            raise ValueError(
                f"Unknown mask '{name}'. "
                f"Available: {list(MASK_REGISTRY.keys())}"
            )

        fn = MASK_REGISTRY[name]

        fn_args = fn.__code__.co_varnames[: fn.__code__.co_argcount]

        call_kwargs = {
            k: v for k, v in kwargs.items()
            if k in fn_args
        }

        partial = fn(seq_len, **call_kwargs)

        if device is not None:
            partial = partial.to(device)

        mask |= partial

    return mask