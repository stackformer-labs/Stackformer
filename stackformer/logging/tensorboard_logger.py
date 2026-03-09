"""
stackformer.logging.tensorboard_logger

TensorBoard logger for StackFormer.

Features
--------
• Logs scalar metrics to TensorBoard
• Compatible with MetricTracker
• Minimal overhead during training
• Step-based logging
• Distributed-safe logging
"""

import os
import time
import datetime

from torch.utils.tensorboard import SummaryWriter

from stackformer.utils.utils import is_main_process, print_once


class TensorBoardLogger:

    def __init__(
        self,
        log_dir="logs",
        experiment_name="stackformer_run",
        auto_timestamp=True,
    ):

        # -------------------------------------------------
        # Distributed safety
        # -------------------------------------------------

        self.enabled = is_main_process()

        if not self.enabled:
            return

        # -------------------------------------------------
        # Experiment folder
        # -------------------------------------------------

        if auto_timestamp:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{experiment_name}_{timestamp}"

        self.log_dir = os.path.join(log_dir, experiment_name)

        os.makedirs(self.log_dir, exist_ok=True)

        # -------------------------------------------------
        # TensorBoard writer
        # -------------------------------------------------

        self.writer = SummaryWriter(self.log_dir)

        self.step = 0
        self.start_time = time.time()

        print_once(f"[TensorBoard] Logging → {self.log_dir}")

    # -----------------------------------------------------

    def log(self, metrics: dict):
        """
        Log metrics to TensorBoard.
        """

        if not self.enabled:
            return

        if not metrics:
            return

        for name, value in metrics.items():

            if value is None:
                continue

            try:

                if isinstance(value, (int, float)):

                    self.writer.add_scalar(name, value, self.step)

            except Exception:
                # Ignore unsupported metric types
                pass

        self.step += 1

    # -----------------------------------------------------

    def flush(self):

        if self.enabled and self.writer:
            self.writer.flush()

    # -----------------------------------------------------

    def close(self):

        if self.enabled and self.writer:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    # -----------------------------------------------------

    def __del__(self):

        try:
            self.close()
        except Exception:
            pass