"""High-level Trainer API for StackFormer V2."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader, Subset

from stackformer.distributed.ddp import init_distributed, is_distributed, wrap_model_ddp
from stackformer.engine.checkpoint import CheckpointManager
from stackformer.engine.engine import Engine
from stackformer.engine.state import TrainingState
from stackformer.optim.factories import create_optimizer, create_scheduler
from stackformer.optim.loss_fn import language_modeling_cross_entropy
from stackformer.utils.device import get_device, print_device_info
from stackformer.utils.utils import is_main_process, print_once, seed_everything


class Trainer:
    """Configurable trainer for research workloads.

    Args:
        model: Model to train.
        train_dataloader: Training dataloader.
        val_dataloader: Validation dataloader.
        optimizer: Optional externally created optimizer.
        scheduler: Optional externally created scheduler.
        criterion: Optional custom loss callable. If ``None``, language-modeling
            cross-entropy is used.
        device: ``"auto"`` or explicit torch device string.
        seed: Optional reproducibility seed.
        max_epochs: Maximum number of epochs.
        max_train_steps: Optional limit on optimizer steps.
        max_eval_steps: Optional limit on validation iterations per epoch.
        eval_every_n_epochs: Validation cadence.
        save_every_n_epochs: Checkpoint cadence.
        grad_accumulation_step: Gradient accumulation factor.
        max_grad_norm: Optional gradient clipping norm.
        checkpoint_dir: Checkpoint directory.
        resume_from: Optional checkpoint path to restore.
        monitor: Optional logging backend object with ``log(dict)``.
        scaler: Optional AMP scaler wrapper.
        use_ddp: If ``True``, initialize and use DistributedDataParallel.
        ddp_backend: Optional torch distributed backend (e.g. ``"nccl"``).
        lr: Learning rate when optimizer is auto-created.
        weight_decay: Weight decay when optimizer is auto-created.
        optimizer_name: Optimizer factory key.
        scheduler_name: Scheduler factory key.
        warmup_steps: Warmup steps for warmup schedulers.
        total_steps: Total steps for step-based schedulers.
        max_steps: Backward-compatible alias for ``max_train_steps``.
        max_eval_step: Backward-compatible alias for ``max_eval_steps``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        criterion: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        device: str = "auto",
        seed: Optional[int] = None,
        max_epochs: int = 1,
        max_train_steps: Optional[int] = None,
        max_eval_steps: Optional[int] = None,
        eval_every_n_epochs: int = 1,
        save_every_n_epochs: int = 1,
        grad_accumulation_step: int = 1,
        max_grad_norm: Optional[float] = None,
        checkpoint_dir: str = "checkpoints",
        resume_from: Optional[str] = None,
        monitor: Optional[Any] = None,
        scaler: Optional[Any] = None,
        use_ddp: bool = False,
        ddp_backend: Optional[str] = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        optimizer_name: str = "adamw",
        scheduler_name: str = "none",
        warmup_steps: int = 0,
        total_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
        max_eval_step: Optional[int] = None,
    ):
        if model is None:
            raise ValueError("`model` must be provided.")

        self.model = model
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or language_modeling_cross_entropy

        self.device = device
        self.seed = seed
        self.max_epochs = int(max_epochs)
        self.max_train_steps = max_train_steps if max_train_steps is not None else max_steps
        self.max_eval_steps = max_eval_steps if max_eval_steps is not None else max_eval_step
        self.eval_every_n_epochs = max(1, int(eval_every_n_epochs))
        self.save_every_n_epochs = max(1, int(save_every_n_epochs))
        self.grad_accumulation_step = max(1, int(grad_accumulation_step))
        self.max_grad_norm = max_grad_norm
        self.monitor = monitor
        self.scaler = scaler
        self.use_ddp = bool(use_ddp)
        self.ddp_backend = ddp_backend
        self.resume_from = resume_from
        self.checkpoint_dir = checkpoint_dir

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        self._setup_seed()
        self._setup_distributed()
        self._setup_device()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_checkpoint_manager()

        self.state = TrainingState(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
            config={"criterion": self.criterion},
        )

        self.engine = Engine(
            state=self.state,
            grad_accum_steps=self.grad_accumulation_step,
            max_grad_norm=self.max_grad_norm,
            monitor=self.monitor,
            max_train_steps=self.max_train_steps,
        )
        self.engine.max_eval_steps = self.max_eval_steps

        if self.resume_from:
            self.load(self.resume_from)

    def _setup_seed(self) -> None:
        if self.seed is not None:
            seed_everything(self.seed)

    def _setup_distributed(self) -> None:
        if not self.use_ddp:
            return

        try:
            init_distributed(backend=self.ddp_backend)
        except Exception as exc:
            print_once(f"[Trainer] Distributed initialization skipped: {exc}")

    def _setup_device(self) -> None:
        if self.device == "auto":
            self.device = get_device()

        self.model.to(self.device)
        if self.use_ddp and is_distributed():
            self.model = wrap_model_ddp(self.model)
        print_device_info()

    def _setup_optimizer(self) -> None:
        if self.optimizer is not None:
            return
        self.optimizer = create_optimizer(
            self.model,
            optimizer_name=self.optimizer_name,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _setup_scheduler(self) -> None:
        if self.scheduler is not None:
            return
        if self.scheduler_name is None or self.scheduler_name == "none":
            self.scheduler = None
            return

        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_name=self.scheduler_name,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
        )

    def _setup_checkpoint_manager(self) -> None:
        self.checkpoint_manager = CheckpointManager(output_dir=self.checkpoint_dir, device=self.device)

    def _build_resume_dataloader(self, dataloader: Optional[DataLoader]) -> Optional[DataLoader]:
        if dataloader is None or self.state.batch_idx == 0:
            return dataloader

        dataset = dataloader.dataset
        batch_size = dataloader.batch_size
        if batch_size is None:
            return dataloader

        start_sample = self.state.batch_idx * batch_size
        if start_sample >= len(dataset):
            return dataloader

        subset = Subset(dataset, range(start_sample, len(dataset)))
        resumed_loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=getattr(dataloader, "num_workers", 0),
            pin_memory=getattr(dataloader, "pin_memory", False),
            drop_last=getattr(dataloader, "drop_last", False),
        )
        print_once(f"[Trainer] Resuming dataloader from sample {start_sample}/{len(dataset)}")
        return resumed_loader

    def fit(self) -> None:
        if self.train_loader is None:
            raise ValueError("train_dataloader is required to call fit().")

        for epoch in range(self.state.epoch, self.max_epochs):
            train_loader = (
                self._build_resume_dataloader(self.train_loader)
                if epoch == self.state.epoch and self.state.batch_idx > 0
                else self.train_loader
            )
            self.engine.train_one_epoch(train_loader, epoch)

            if self.engine.reached_max_train_steps():
                if is_main_process():
                    self.save("latest")
                break

            if self.val_loader is not None and ((epoch + 1) % self.eval_every_n_epochs == 0):
                self.engine.validate_one_epoch(self.val_loader, epoch)

            self.state.reset_batch()
            self.state.next_epoch()

            if is_main_process() and ((epoch + 1) % self.save_every_n_epochs == 0):
                self.save("latest")

    def validate(self) -> None:
        if self.val_loader is None:
            raise ValueError("Validation dataloader not provided.")
        self.engine.validate_one_epoch(self.val_loader, self.state.epoch)

    def save(self, name: str = "latest", export_jit: bool = False) -> None:
        jit_path = None
        if export_jit:
            try:
                jit_path = self.checkpoint_manager.save_jit_model(self.model, f"{name}_model")
            except Exception as exc:
                print_once(f"[Trainer] TorchScript export skipped: {exc}")

        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "scaler": self.scaler,
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "batch_idx": self.state.batch_idx,
            "jit_model_path": jit_path,
        }
        self.checkpoint_manager.save(state, name)

    def load(self, checkpoint_path: str) -> None:
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "scaler": self.scaler,
        }
        metadata = self.checkpoint_manager.load(checkpoint_path, state)
        self.state.load_metadata(metadata)
        print_once(f"[Trainer] Resumed from {checkpoint_path}")

    def export_torchscript(self, name: str = "inference_model") -> str:
        """Export current model as TorchScript artifact and return path."""
        return self.checkpoint_manager.save_jit_model(self.model, name=name)
