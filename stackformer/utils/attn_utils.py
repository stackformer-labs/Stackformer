"""Attention module helpers."""

from collections import OrderedDict

import torch
import torch.nn.functional as F

from stackformer.modules.Masking import make_mask

# Maximum number of cached attention masks.
_MAX_MASK_CACHE_SIZE = 32


def _run_sdpa(q, k, v, attn_mask, dropout_p,):
    """
    Wrapper around PyTorch's scaled dot-product attention (SDPA).

    This function standardizes how attention is executed across the codebase.

    Args:
        q (Tensor): Query tensor of shape (B, H, T, D)
        k (Tensor): Key tensor of shape (B, H, T, D)
        v (Tensor): Value tensor of shape (B, H, T, D)
        attn_mask (Tensor or None): Attention mask.
        dropout_p (float): Attention dropout probability.

    Returns:
        Tensor: Attention output.
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
    Normalize user-provided mask_type.

    Args:
        mask_type:
            True           -> ["causal"]
            False / None   -> None
            str            -> [str]
            list / tuple   -> list

    Returns:
        list[str] | None
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


def _canonical_device(device) -> torch.device:
    """
    Convert a device specification into a canonical torch.device.

    This ensures cache keys remain consistent between
    'cuda' and 'cuda:0'.
    """
    device = torch.device(device)

    if device.type == "cuda" and device.index is None:
        current = torch.cuda.current_device() if torch.cuda.is_available() else 0
        device = torch.device(f"cuda:{current}")

    return device


def _get_attention_mask(
    cache: OrderedDict,
    mask_type,
    seq_len: int,
    device,
    **mask_kwargs,
):
    """
    Retrieve or create a cached attention mask.

    The cache is bounded using FIFO eviction to prevent
    unbounded memory growth.

    Args:
        cache (OrderedDict):
            Attention mask cache.
        mask_type:
            Mask specification.
        seq_len (int):
            Sequence length.
        device:
            Torch device.
        **mask_kwargs:
            Additional arguments forwarded to make_mask().

    Returns:
        Tensor | None
    """
    mask_types = _normalize_mask_type(mask_type)

    if mask_types is None:
        return None

    device = _canonical_device(device)

    key = (
        seq_len,
        tuple(mask_types),
        device,
    )

    if key not in cache:

        if len(cache) >= _MAX_MASK_CACHE_SIZE:
            cache.popitem(last=False)

        cache[key] = make_mask(
            mask_types,
            seq_len,
            device=device,
            **mask_kwargs,
        )

    return cache[key]