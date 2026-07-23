"""
masks.py
--------

Attention mask utilities for dense and sparse transformer layouts.

All masks follow the PyTorch SDPA boolean convention:

    True  -> visible (token takes part in attention)
    False -> masked  (token is ignored, -inf added to logit)

Returned shape:
    (seq_len, seq_len)   —  row = query index, col = key index

This module supports composing multiple masking strategies using
a registry-based factory pattern with configurable composition operators.
"""

from typing import Callable, Dict, List, Literal
import torch
import inspect

# Base Mask Functions
def causal(seq_len: int, device=None) -> torch.Tensor:
    """
    Standard autoregressive causal mask.

    Each token attends to itself and all previous tokens.
    Future tokens are masked.

    Args:
        seq_len (int): Sequence length.

    Returns:
        torch.Tensor: (seq_len, seq_len) boolean mask.
            True  = visible (lower triangle, diagonal included)
            False = masked  (upper triangle, future tokens)
    """
    return torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )


def sliding_window(seq_len: int, window_size: int, device=None) -> torch.Tensor:
    """
    Sliding window causal attention.

    Each token attends to itself and the previous ``window_size - 1``
    tokens. Tokens further in the past and all future tokens are masked.

    Args:
        seq_len (int)
        window_size (int): Number of positions each token can see,
            including itself.  E.g. window_size=3 → self + 2 past.

    Returns:
        torch.Tensor: (seq_len, seq_len) boolean mask.
            True  = visible
            False = masked
    """
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    
    i = torch.arange(seq_len, device=device).unsqueeze(1)   # (seq_len, 1) — query index
    j = torch.arange(seq_len, device=device).unsqueeze(0)   # (1, seq_len) — key index

    in_past   = j <= i                        # key is not a future token
    in_window = (i - j) < window_size         # key is within the window
    
    return in_past & in_window


def dilated_causal(seq_len: int, dilation: int, device=None) -> torch.Tensor:
    """
    Dilated causal attention.

    Each token attends only to past positions that are exact multiples
    of ``dilation`` steps away, including itself (distance 0).
    All other positions are masked.

    Args:
        seq_len (int)
        dilation (int)

    Returns:
        torch.Tensor: (seq_len, seq_len) boolean mask.
            True  = visible
            False = masked
    """
    if dilation <= 0:
        raise ValueError("dilation must be > 0")
 
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)

    in_past   = j <= i
    on_stride = (i - j) % dilation == 0
    return in_past & on_stride


def random_mask(seq_len: int, num_random: int, device=None) -> torch.Tensor:
    """
    Random sparse causal attention.

    Each token attends to ``num_random`` randomly-chosen positions from
    its causal past (including itself). All other positions are masked.

    Args:
        seq_len (int)
        num_random (int)

    Returns:
        torch.Tensor: (seq_len, seq_len) boolean mask.
            True  = visible
            False = masked
    """
    if num_random < 0 or num_random > seq_len:
        raise ValueError("num_random must be between 0 and seq_len")
    # Start fully masked; selectively open chosen past positions.
    cols = (
        torch.rand(seq_len, seq_len, device=device)
        .tril()
        .topk(num_random, dim=1)
        .indices
    )

    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    mask.scatter_(1, cols, True)
    return mask


def global_mask(seq_len: int, global_index: List[int], device=None) -> torch.Tensor:
    """
    Global attention mask.

    Selected global tokens attend to every other token, and every other
    token attends back to the global tokens.
    All remaining pairs are masked.

    Args:
        seq_len (int)
        global_index (List[int]): Token indices with global visibility.

    Returns:
        torch.Tensor: (seq_len, seq_len) boolean mask.
            True  = visible
            False = masked
    """
    if any(i < 0 or i >= seq_len for i in global_index):
        raise ValueError("global_index contains invalid token indices")
    # Start fully masked; open global rows and columns.
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    g    = torch.tensor(global_index, device=device)

    mask[g, :] = True   # global tokens → can attend to everyone
    mask[:, g] = True   # everyone      → can attend to global tokens

    return mask


def no_mask(seq_len: int, device=None) -> torch.Tensor:
    """
    Full bidirectional attention — every position is visible.

    Args:
        seq_len (int)

    Returns:
        torch.Tensor: (seq_len, seq_len) boolean mask, all True.
    """
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)


def mistral(seq_len: int, window_size: int, dilation: int, device=None) -> torch.Tensor:
    """
    Mistral-style hybrid mask: sliding window + dilated attention.

    A position is visible if it falls inside the sliding window OR on
    the dilated stride. Positions outside both are masked.

    Args:
        seq_len (int)
        window_size (int)
        dilation (int)

    Returns:
        torch.Tensor: (seq_len, seq_len) boolean mask.
            True  = visible
            False = masked
    """
    return (sliding_window(seq_len, window_size, device=device)| dilated_causal(seq_len, dilation, device=device))

# Registry
MASK_REGISTRY: Dict[str, Callable] = {
    "no":             no_mask,
    "causal":         causal,
    "sliding_window": sliding_window,
    "dilated_causal": dilated_causal,
    "random_mask":    random_mask,
    "global_mask":    global_mask,
    "mistral":        mistral,
}

# Unified Factory

def make_mask(
    mask_types: List[str],
    seq_len: int,
    device: torch.device | None = None,
    combine: Literal["or", "and"] = "or",
    **kwargs,
) -> torch.Tensor:
    """
    Unified mask factory.

    Combines multiple mask patterns into a single boolean mask using
    the True = visible / False = masked convention.

    Args:
        mask_types (List[str]):
            Mask names to combine. See ``MASK_REGISTRY`` for available keys.

        seq_len (int):
            Sequence length.

        device (torch.device, optional):
            Target device for the returned tensor.

        combine (Literal["or", "and"]):
            How to merge individual masks.

            ``"or"``  — a position is visible if ANY sub-mask marks it
                        visible (union of visible sets; less restrictive).
                        Identity: all-False.  Accumulates with ``|``.

            ``"and"`` — a position is visible only if ALL sub-masks mark
                        it visible (intersection; more restrictive).
                        Identity: all-True.  Accumulates with ``&``.

            Default: ``"or"``.

        **kwargs:
            Extra parameters forwarded to mask functions by name::

                window_size  (int)       — sliding_window, mistral
                dilation     (int)       — dilated_causal, mistral
                num_random   (int)       — random_mask
                global_index (List[int]) — global_mask

    Returns:
        torch.Tensor:
            Boolean mask of shape ``(seq_len, seq_len)``.
            True = visible, False = masked.

    Raises:
        TypeError:  if ``mask_types`` is not a list or tuple.
        ValueError: if an unknown mask name is given, or ``combine`` is
                    not ``"or"`` or ``"and"``.

    Examples::

        # Causal OR global — visible if causal past OR is a global token
        mask = make_mask(
            ["causal", "global_mask"],
            seq_len=16,
            global_index=[0],
            combine="or",
        )

        # Sliding window AND dilated — must be visible in both to attend
        mask = make_mask(
            ["sliding_window", "dilated_causal"],
            seq_len=32,
            window_size=4,
            dilation=2,
            combine="and",
        )
    """
    if not isinstance(mask_types, (list, tuple)):
        raise TypeError("mask_types must be a list of strings")

    if combine not in ("or", "and"):
        raise ValueError(f"combine must be 'or' or 'and', got {combine!r}")

    # Identity element:
    #   OR  → all-False (nothing visible yet); grow with |
    #   AND → all-True  (everything visible); restrict with &
    init_val = combine == "and"
    mask = torch.full(
        (seq_len, seq_len),
        fill_value=init_val,
        dtype=torch.bool,
        device=device,
    )

    for name in mask_types:
        if name not in MASK_REGISTRY:
            raise ValueError(
                f"Unknown mask '{name}'. "
                f"Available: {list(MASK_REGISTRY.keys())}"
            )
            
        fn  = MASK_REGISTRY[name]
        sig = inspect.signature(fn)
        call_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in sig.parameters
        }

        partial = fn(seq_len, device=device, **call_kwargs)

        if combine == "or":
            mask |= partial
        else:
            mask &= partial

    return mask