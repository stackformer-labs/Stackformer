"""
stackformer.optim.factories

Optimizer and scheduler factories.

Supports:
- LLM / Transformer training
- Vision models (ViT, CNNs)
- Segmentation models (SegFormer, UNet)
- General PyTorch models
"""

from __future__ import annotations

import math
import torch
from torch.optim import Adam, AdamW, SGD, RMSprop, Adagrad
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    ReduceLROnPlateau,
    LambdaLR,
)

# -----------------------------------------------------
# Optimizer Registry
# -----------------------------------------------------

OPTIMIZERS = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
    "rmsprop": RMSprop,
    "adagrad": Adagrad,
}

# -----------------------------------------------------
# Scheduler Registry
# -----------------------------------------------------

SCHEDULERS = {
    "step": StepLR,
    "cosine": CosineAnnealingLR,
    "cosine_restart": CosineAnnealingWarmRestarts,
    "exponential": ExponentialLR,
    "plateau": ReduceLROnPlateau,
}

# -----------------------------------------------------
# Parameter Group Helpers
# -----------------------------------------------------

def get_parameter_groups(model, weight_decay: float = 0.0):
    """
    Create parameter groups separating parameters that should
    and should not use weight decay.

    Important for:
    - Transformers
    - LLM training
    - Vision Transformers
    """

    decay = []
    no_decay = []

    whitelist_weight_modules = (
        torch.nn.Linear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
    )

    blacklist_weight_modules = (
        torch.nn.LayerNorm,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.Embedding,
    )

    for module_name, module in model.named_modules():

        for param_name, param in module.named_parameters(recurse=False):

            if not param.requires_grad:
                continue

            full_name = f"{module_name}.{param_name}" if module_name else param_name

            if param_name.endswith("bias"):
                no_decay.append(param)

            elif isinstance(module, whitelist_weight_modules):
                decay.append(param)

            elif isinstance(module, blacklist_weight_modules):
                no_decay.append(param)

            else:
                decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# -----------------------------------------------------
# Optimizer Factory
# -----------------------------------------------------

def create_optimizer(
    model,
    optimizer_name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    momentum: float = 0.9,
):
    """
    Create optimizer.

    Works well for:
    - GPT / LLM
    - BERT / Transformers
    - Vision Transformers
    - SegFormer
    - CNN models
    """

    optimizer_name = optimizer_name.lower()

    if optimizer_name not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    params = get_parameter_groups(model, weight_decay)

    if optimizer_name == "sgd":

        optimizer = SGD(
            params,
            lr=lr,
            momentum=momentum,
        )

    else:

        optimizer = OPTIMIZERS[optimizer_name](
            params,
            lr=lr,
        )

    return optimizer


# -----------------------------------------------------
# Warmup Schedulers (important for Transformers)
# -----------------------------------------------------

def linear_warmup_scheduler(optimizer, warmup_steps, total_steps):

    if total_steps is None:
        raise ValueError("total_steps must be provided for warmup schedulers")

    def lr_lambda(step):

        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        return max(
            0.0,
            float(total_steps - step)
            / float(max(1, total_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps):

    if total_steps is None:
        raise ValueError("total_steps must be provided for warmup schedulers")

    def lr_lambda(step):

        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )

        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# -----------------------------------------------------
# Scheduler Factory
# -----------------------------------------------------

def create_scheduler(
    optimizer,
    scheduler_name: str = "cosine",
    total_steps: int | None = None,
    warmup_steps: int = 0,
    step_size: int = 10,
    gamma: float = 0.1,
):
    """
    Create LR scheduler.

    Common choices:

    LLM / Transformers
        cosine_warmup
        linear_warmup

    Vision / Segmentation
        cosine
        step

    General
        exponential
        plateau
    """

    scheduler_name = scheduler_name.lower()

    # ---------------------------------------------
    # Transformer schedulers
    # ---------------------------------------------

    if scheduler_name == "linear_warmup":

        return linear_warmup_scheduler(
            optimizer,
            warmup_steps,
            total_steps,
        )

    if scheduler_name == "cosine_warmup":

        return cosine_warmup_scheduler(
            optimizer,
            warmup_steps,
            total_steps,
        )

    # ---------------------------------------------
    # Standard schedulers
    # ---------------------------------------------

    if scheduler_name == "cosine":

        if total_steps is None:
            raise ValueError("total_steps required for cosine scheduler")

        return CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
        )

    if scheduler_name == "cosine_restart":

        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=step_size,
        )

    if scheduler_name == "step":

        return StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    if scheduler_name == "exponential":

        return ExponentialLR(
            optimizer,
            gamma=gamma,
        )

    if scheduler_name == "plateau":

        return ReduceLROnPlateau(
            optimizer,
            factor=gamma,
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")