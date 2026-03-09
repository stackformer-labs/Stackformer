"""
stackformer.logging.logger

Unified multi-backend logger for StackFormer.

Supports:
---------
• CSV logging
• TensorBoard logging
• Weights & Biases logging

Design goals
------------
• Single interface for all logging backends
• Minimal overhead
• Easy to extend
• Compatible with Trainer monitor system
"""

from stackformer.logging.csv_logger import CSVLogger
from stackformer.logging.tensorboard_logger import TensorBoardLogger
from stackformer.utils.utils import print_once


class Logger:

    def __init__(
        self,
        csv=True,
        tensorboard=False,
        wandb=False,
        log_dir="logs",
        experiment_name="run",
        wandb_project="stackformer",
        wandb_config=None,
    ):

        self.backends = []

        # --------------------------------
        # CSV Logger
        # --------------------------------

        if csv:

            self.backends.append(
                CSVLogger(
                    log_dir=log_dir,
                    filename=f"{experiment_name}_metrics.csv",
                )
            )

            print_once("[Logger] CSV logging enabled")

        # --------------------------------
        # TensorBoard Logger
        # --------------------------------

        if tensorboard:

            self.backends.append(
                TensorBoardLogger(
                    log_dir=log_dir,
                    experiment_name=experiment_name,
                )
            )

            print_once("[Logger] TensorBoard logging enabled")

        # --------------------------------
        # WandB Logger
        # --------------------------------

        if wandb:

            try:
                from stackformer.logging.wandb_logger import WandBLogger

                self.backends.append(
                    WandBLogger(
                        project=wandb_project,
                        experiment_name=experiment_name,
                        config=wandb_config,
                    )
                )

                print_once("[Logger] Weights & Biases logging enabled")

            except ImportError:

                raise ImportError(
                    "wandb is not installed. Install with `pip install wandb`."
                )

        if not self.backends:
            print_once("[Logger] No logging backend enabled")

    # -----------------------------------------------------

    def log(self, metrics: dict):
        """
        Log metrics to all active backends.
        """

        if not metrics:
            return

        for backend in self.backends:

            try:
                backend.log(metrics)
            except Exception:
                # prevent training interruption
                pass

    # -----------------------------------------------------

    def close(self):
        """
        Close all loggers gracefully.
        """

        for backend in self.backends:

            try:

                if hasattr(backend, "flush"):
                    backend.flush()

                if hasattr(backend, "close"):
                    backend.close()

                if hasattr(backend, "finish"):
                    backend.finish()

            except Exception:
                pass

    # -----------------------------------------------------

    def __del__(self):

        try:
            self.close()
        except Exception:
            pass