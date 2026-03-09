"""
stackformer.logging.wandb_logger

Weights & Biases logger for StackFormer.

Features
--------
• Logs training metrics to W&B
• Compatible with StackFormer monitor interface
• Automatic experiment creation
• Supports config logging
• Distributed-safe logging
"""

import time

from stackformer.utils.utils import is_main_process, print_once


class WandBLogger:

    def __init__(
        self,
        project="stackformer",
        experiment_name=None,
        config=None,
        entity=None,
        watch_model=False,
    ):

        # -------------------------------------------------
        # Distributed safety
        # -------------------------------------------------

        self.enabled = is_main_process()

        if not self.enabled:
            return

        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is not installed. Install with: pip install wandb"
            )

        self.wandb = wandb

        # -------------------------------------------------
        # Initialize run
        # -------------------------------------------------

        self.run = self.wandb.init(
            project=project,
            name=experiment_name,
            entity=entity,
            config=config,
        )

        self.step = 0
        self.start_time = time.time()
        self.watch_model_enabled = watch_model

        print_once(f"[WandB] Logging run → {self.run.name}")

    # -----------------------------------------------------

    def log(self, metrics: dict):
        """
        Log metrics to Weights & Biases.
        """

        if not self.enabled:
            return

        if not metrics:
            return

        metrics = dict(metrics)

        metrics["step"] = self.step
        metrics["time"] = time.time() - self.start_time

        # Filter non-numeric values
        clean_metrics = {}

        for k, v in metrics.items():

            if isinstance(v, (int, float)):
                clean_metrics[k] = v

        try:
            self.wandb.log(clean_metrics, step=self.step)
        except Exception:
            pass

        self.step += 1

    # -----------------------------------------------------

    def log_model(self, model):
        """
        Optional: watch gradients and parameters.
        """

        if not self.enabled:
            return

        if not self.watch_model_enabled:
            return

        try:
            self.wandb.watch(model)
        except Exception:
            pass

    # -----------------------------------------------------

    def finish(self):
        """
        Finish W&B run.
        """

        if not self.enabled:
            return

        try:
            if self.run is not None:
                self.wandb.finish()
        except Exception:
            pass

    # -----------------------------------------------------

    def __del__(self):

        try:
            self.finish()
        except Exception:
            pass