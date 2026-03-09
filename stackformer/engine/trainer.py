import torch
from torch.utils.data import DataLoader, Subset

from stackformer.engine.engine import Engine
from stackformer.engine.state import TrainingState
from stackformer.engine.checkpoint import CheckpointManager

from stackformer.optim.factories import create_optimizer, create_scheduler

from stackformer.utils.utils import (
    seed_everything,
    print_once,
    is_main_process,
)

from stackformer.utils.device import get_device, print_device_info
from stackformer.distributed.ddp import (
    init_distributed,
    wrap_model_ddp,
    is_distributed,
)


class Trainer:
    """
    High-level training interface for StackFormer.

    Responsibilities
    ----------------
    • Setup training environment
    • Initialize optimizer / scheduler
    • Manage checkpoints
    • Handle resume training
    • Delegate execution to Engine
    """

    def __init__(
        self,
        model,
        train_dataloader=None,
        val_dataloader=None,
        optimizer=None,
        scheduler=None,
        criterion=None,
        device="auto",
        seed=None,
        max_epochs=1,
        checkpoint_dir="checkpoints",
        resume_from=None,
        monitor=None,
        scaler=None,

        # optimizer / scheduler config
        lr=3e-4,
        weight_decay=0.01,
        optimizer_name="adamw",
        scheduler_name="none",
        warmup_steps=0,
        total_steps=None,
    ):

        # -------------------------------------------------
        # Core objects
        # -------------------------------------------------

        self.model = model
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.device = device
        self.seed = seed
        self.max_epochs = max_epochs

        self.monitor = monitor
        self.scaler = scaler

        self.resume_from = resume_from
        self.checkpoint_dir = checkpoint_dir

        # -------------------------------------------------
        # Optimization config
        # -------------------------------------------------

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        # -------------------------------------------------
        # Environment setup
        # -------------------------------------------------

        self._setup_seed()
        self._setup_distributed()
        self._setup_device()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_checkpoint_manager()

        # -------------------------------------------------
        # Training state
        # -------------------------------------------------

        self.state = TrainingState(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
            config={"criterion": self.criterion},
        )

        # -------------------------------------------------
        # Runtime engine
        # -------------------------------------------------

        self.engine = Engine(
            state=self.state,
            monitor=self.monitor,
        )

        # -------------------------------------------------
        # Resume training
        # -------------------------------------------------

        if self.resume_from:
            self.load(self.resume_from)

    # -----------------------------------------------------
    # Seed setup
    # -----------------------------------------------------

    def _setup_seed(self):

        if self.seed is not None:
            seed_everything(self.seed)

    # -----------------------------------------------------
    # Distributed setup
    # -----------------------------------------------------

    def _setup_distributed(self):

        try:
            init_distributed()
        except Exception:
            pass

    # -----------------------------------------------------
    # Device setup
    # -----------------------------------------------------

    def _setup_device(self):

        if self.device == "auto":
            self.device = get_device()

        self.model.to(self.device)

        # Wrap model if running distributed
        if is_distributed():
            self.model = wrap_model_ddp(self.model)

        print_device_info()

    # -----------------------------------------------------
    # Optimizer setup
    # -----------------------------------------------------

    def _setup_optimizer(self):

        if self.optimizer is None:

            self.optimizer = create_optimizer(
                self.model,
                optimizer_name=self.optimizer_name,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

    # -----------------------------------------------------
    # Scheduler setup
    # -----------------------------------------------------

    def _setup_scheduler(self):

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

    # -----------------------------------------------------
    # Checkpoint manager
    # -----------------------------------------------------

    def _setup_checkpoint_manager(self):

        self.checkpoint_manager = CheckpointManager(
            output_dir=self.checkpoint_dir,
            device=self.device,
        )

    # -----------------------------------------------------
    # Resume dataloader
    # -----------------------------------------------------

    def _build_resume_dataloader(self, dataloader):

        if dataloader is None:
            return None

        if self.state.batch_idx == 0:
            return dataloader

        dataset = dataloader.dataset
        batch_size = dataloader.batch_size

        start_sample = self.state.batch_idx * batch_size

        if start_sample >= len(dataset):
            return dataloader

        subset_indices = range(start_sample, len(dataset))
        subset = Subset(dataset, subset_indices)

        resumed_loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=getattr(dataloader, "num_workers", 0),
            pin_memory=getattr(dataloader, "pin_memory", False),
            drop_last=getattr(dataloader, "drop_last", False),
        )

        print_once(
            f"[Trainer] Resuming dataloader from sample {start_sample}/{len(dataset)}"
        )

        return resumed_loader

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    def fit(self):

        for epoch in range(self.state.epoch, self.max_epochs):

            if epoch == self.state.epoch and self.state.batch_idx > 0:
                train_loader = self._build_resume_dataloader(self.train_loader)
            else:
                train_loader = self.train_loader

            self.engine.train_one_epoch(train_loader, epoch)

            if self.val_loader is not None:
                self.engine.validate_one_epoch(self.val_loader, epoch)

            self.state.reset_batch()
            self.state.next_epoch()

            # Save checkpoint each epoch
            if is_main_process():
                self.save("latest")

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    def validate(self):

        if self.val_loader is None:
            raise ValueError("Validation dataloader not provided.")

        self.engine.validate_one_epoch(self.val_loader, self.state.epoch)

    # -----------------------------------------------------
    # Save checkpoint
    # -----------------------------------------------------

    def save(self, name="latest"):

        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "scaler": self.scaler,
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "batch_idx": self.state.batch_idx,
        }

        self.checkpoint_manager.save(state, name)

    # -----------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------

    def load(self, checkpoint_path):

        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "scaler": self.scaler,
        }

        metadata = self.checkpoint_manager.load(checkpoint_path, state)

        self.state.load_metadata(metadata)

        print_once(f"[Trainer] Resumed from {checkpoint_path}")