"""Loss function helpers for Transformer-family training.

The project uses cross-entropy as the default objective for language modeling,
but researchers often need additional losses for multitask, contrastive,
classification, and segmentation workloads.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def language_modeling_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Compute autoregressive token cross-entropy loss.

    Args:
        logits: Tensor of shape ``[B, T, V]``.
        labels: Tensor of shape ``[B, T]``.
        ignore_index: Label id to skip.
        label_smoothing: Smoothing factor in ``[0, 1)``.

    Returns:
        Scalar loss tensor.
    """
    if logits.ndim != 3:
        raise ValueError("Expected logits shape [B, T, V] for language modeling.")
    if labels.ndim != 2:
        raise ValueError("Expected labels shape [B, T] for language modeling.")

    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )


def classification_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Compute cross-entropy loss for classifier heads.

    Args:
        logits: Tensor of shape ``[B, C]``.
        labels: Tensor of shape ``[B]``.
        label_smoothing: Smoothing factor.
    """
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


def binary_classification_bce_with_logits(logits: Tensor, labels: Tensor) -> Tensor:
    """Compute BCE-with-logits for binary/multi-label tasks."""
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def segmentation_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = 255,
) -> Tensor:
    """Compute segmentation cross-entropy.

    Args:
        logits: Tensor ``[B, C, H, W]``.
        labels: Tensor ``[B, H, W]``.
        ignore_index: Masked class id.
    """
    return F.cross_entropy(logits, labels.long(), ignore_index=ignore_index)


def kl_divergence_distillation(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """KL divergence distillation loss.

    Useful for student/teacher Transformer training.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


def get_loss_fn(name: str, **kwargs) -> Callable[[Tensor, Tensor], Tensor]:
    """Factory for common loss functions.

    Supported names:
        - ``lm_cross_entropy``
        - ``classification_cross_entropy``
        - ``bce_with_logits``
        - ``segmentation_cross_entropy``
    """
    key = name.lower()

    if key == "lm_cross_entropy":
        return lambda logits, labels: language_modeling_cross_entropy(logits, labels, **kwargs)
    if key == "classification_cross_entropy":
        return lambda logits, labels: classification_cross_entropy(logits, labels, **kwargs)
    if key == "bce_with_logits":
        return lambda logits, labels: binary_classification_bce_with_logits(logits, labels)
    if key == "segmentation_cross_entropy":
        return lambda logits, labels: segmentation_cross_entropy(logits, labels, **kwargs)

    raise ValueError(f"Unsupported loss function: {name}")
