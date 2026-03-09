import torch
from tqdm import tqdm

from stackformer.utils.utils import is_main_process


# -----------------------------------------------------
# Training loop
# -----------------------------------------------------

def train_loop(engine, dataloader, epoch):

    state = engine.state

    state.model.train()

    # -------------------------------------------------
    # Distributed sampler epoch reset (important)
    # -------------------------------------------------

    sampler = getattr(dataloader, "sampler", None)

    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    # -------------------------------------------------
    # Iterator setup
    # -------------------------------------------------

    iterator = enumerate(dataloader)

    show_progress = not engine.live_monitoring and is_main_process()

    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(dataloader),
            desc=f"Train Epoch {epoch}",
        )

    # -------------------------------------------------
    # Training iteration
    # -------------------------------------------------

    for batch_idx, batch in iterator:

        metrics = engine._train_step(batch)

        # -------------------------------------------------
        # tqdm display (rank 0 only)
        # -------------------------------------------------

        if show_progress and metrics:

            display_metrics = {}

            if "loss" in metrics:
                display_metrics["loss"] = f"{metrics['loss']:.4f}"

            if "lr" in metrics and metrics["lr"] is not None:
                display_metrics["lr"] = f"{metrics['lr']:.2e}"

            if "step_time" in metrics:
                display_metrics["step"] = f"{metrics['step_time']:.3f}s"

            iterator.set_postfix(display_metrics)


# -----------------------------------------------------
# Validation loop
# -----------------------------------------------------

def validation_loop(engine, dataloader, epoch):

    state = engine.state

    state.model.eval()

    # -------------------------------------------------
    # Iterator setup
    # -------------------------------------------------

    iterator = enumerate(dataloader)

    show_progress = not engine.live_monitoring and is_main_process()

    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(dataloader),
            desc=f"Validation Epoch {epoch}",
        )

    # -------------------------------------------------
    # Evaluation iteration
    # -------------------------------------------------

    with torch.no_grad():

        for batch_idx, batch in iterator:

            metrics = engine._eval_step(batch)

            if show_progress and metrics:

                display_metrics = {}

                if "val_loss" in metrics:
                    display_metrics["val_loss"] = f"{metrics['val_loss']:.4f}"

                iterator.set_postfix(display_metrics)