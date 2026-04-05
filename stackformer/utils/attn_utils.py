"""Attention module helpers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stackformer.modules.Masking import make_mask


def _run_sdpa(q, k, v, attn_mask, dropout_p):
    """
    Wrapper around PyTorch's scaled dot-product attention (SDPA).

    This function standardizes how attention is executed across the codebase.

    Args:
        q (Tensor): Query tensor of shape (B, H, T, D)
        k (Tensor): Key tensor of shape (B, H, T, D)
        v (Tensor): Value tensor of shape (B, H, T, D)
        attn_mask (Tensor or None): Attention mask (broadcastable to SDPA format)
        dropout_p (float): Dropout probability applied to attention weights

    Returns:
        Tensor: Attention output of shape (B, H, T, D)

    Notes:
        - Uses torch.nn.functional.scaled_dot_product_attention
        - `is_causal=False` because masking is handled explicitly via `attn_mask`
        - Keeps behavior consistent across different attention implementations
    """
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
    )


def _normalize_mask_type(mask_type):
    """
    Normalize user-provided mask_type into a standard list format.

    This allows flexible user inputs while keeping internal logic consistent.

    Args:
        mask_type (bool | str | list[str] | None):
            - True  -> ["causal"]
            - False/None -> None (no mask)
            - str   -> [str]
            - list/tuple -> list

    Returns:
        list[str] or None: Normalized mask types

    Raises:
        TypeError: If mask_type is of unsupported type

    Example:
        >>> _normalize_mask_type(True)
        ["causal"]

        >>> _normalize_mask_type("sliding_window")
        ["sliding_window"]
    """
    if mask_type is True:
        return ["causal"]
    if mask_type in (False, None):
        return None
    if isinstance(mask_type, str):
        return [mask_type]
    if isinstance(mask_type, (list, tuple)):
        return list(mask_type)
    raise TypeError("mask_type must be bool, str, or list of str")


def _get_attention_mask(cache: dict, mask_type, seq_len: int, device, **mask_kwargs):
    """
    Retrieve or create an attention mask with caching.

    Masks are expensive to construct repeatedly, especially for large sequence lengths.
    This function caches masks based on configuration to avoid recomputation.

    Args:
        cache (dict): Dictionary used to store previously created masks
        mask_type (bool | str | list[str] | None): Type of mask(s) to apply
        seq_len (int): Sequence length
        device (torch.device): Device on which the mask should be created
        **mask_kwargs: Additional arguments passed to `make_mask`

    Returns:
        Tensor or None: Attention mask tensor or None if no masking is required

    Notes:
        - Cache key includes:
            (seq_len, mask_types, device)
        - Ensures masks are created on the correct device
        - Prevents redundant computation across forward passes

    Example:
        >>> mask = _get_attention_mask(cache, "causal", 128, x.device)
    """
    mask_types = _normalize_mask_type(mask_type)
    if mask_types is None:
        return None

    key = (seq_len, tuple(mask_types), str(device))

    if key not in cache:
        cache[key] = make_mask(
            mask_types,
            seq_len,
            device=device,
            **mask_kwargs
        )

    return cache[key]