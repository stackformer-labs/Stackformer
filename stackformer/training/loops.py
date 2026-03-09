"""Reusable training/evaluation loop helpers."""

from __future__ import annotations

import torch
from tqdm import tqdm

from stackformer.utils.utils import is_main_process


def train_loop(engine, dataloader, epoch: int) -> None:
    """Run one training epoch."""
    if dataloader is None:
        raise ValueError("train dataloader must not be None")

    state = engine.state
    state.model.train()

    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    iterator = enumerate(dataloader)
    show_progress = not engine.live_monitoring and is_main_process()
    if show_progress:
        iterator = tqdm(iterator, total=len(dataloader), desc=f"Train Epoch {epoch}")

    for batch_idx, batch in iterator:
        if engine.reached_max_train_steps():
            break

        metrics = engine._train_step(batch)
        if show_progress and metrics:
            display_metrics = {}
            if "loss" in metrics:
                display_metrics["loss"] = f"{metrics['loss']:.4f}"
            if "lr" in metrics and metrics["lr"] is not None:
                display_metrics["lr"] = f"{metrics['lr']:.2e}"
            if "step_time" in metrics:
                display_metrics["step"] = f"{metrics['step_time']:.3f}s"
            iterator.set_postfix(display_metrics)


def validation_loop(engine, dataloader, epoch: int) -> None:
    """Run one validation epoch."""
    if dataloader is None:
        raise ValueError("validation dataloader must not be None")

    state = engine.state
    state.model.eval()

    iterator = enumerate(dataloader)
    show_progress = not engine.live_monitoring and is_main_process()
    if show_progress:
        iterator = tqdm(iterator, total=len(dataloader), desc=f"Validation Epoch {epoch}")

    max_eval_steps = getattr(engine, "max_eval_steps", None)
    with torch.no_grad():
        for batch_idx, batch in iterator:
            if max_eval_steps is not None and batch_idx >= int(max_eval_steps):
                break

            metrics = engine._eval_step(batch)
            if show_progress and metrics and "val_loss" in metrics:
                iterator.set_postfix({"val_loss": f"{metrics['val_loss']:.4f}"})
