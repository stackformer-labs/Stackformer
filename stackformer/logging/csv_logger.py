"""
stackformer.logging.csv_logger

Lightweight CSV logger for training metrics.

Design goals
------------
• Zero external dependencies
• Minimal overhead during training
• Easy analysis with pandas
• Compatible with MetricTracker
• Distributed-safe logging
"""

import csv
import os
import time

from stackformer.utils.utils import is_main_process, print_once


class CSVLogger:

    def __init__(
        self,
        log_dir="logs",
        filename="metrics.csv",
        flush_every=10,
    ):

        self.enabled = is_main_process()

        if not self.enabled:
            return

        os.makedirs(log_dir, exist_ok=True)

        self.filepath = os.path.join(log_dir, filename)

        self.flush_every = flush_every
        self._step = 0

        self.file = open(self.filepath, "a", newline="")
        self.writer = None
        self.fieldnames = None

        self.start_time = time.time()

        print_once(f"[CSVLogger] Logging metrics → {self.filepath}")

    # --------------------------------------------------

    def log(self, metrics: dict):
        """
        Log metrics for current step.
        """

        if not self.enabled:
            return

        if not metrics:
            return

        metrics = dict(metrics)

        metrics["time"] = time.time() - self.start_time
        metrics["step"] = self._step

        # --------------------------------------------------
        # Initialize writer
        # --------------------------------------------------

        if self.writer is None:

            self.fieldnames = list(metrics.keys())

            self.writer = csv.DictWriter(
                self.file,
                fieldnames=self.fieldnames,
            )

            if self.file.tell() == 0:
                self.writer.writeheader()

        # --------------------------------------------------
        # Handle new metric keys dynamically
        # --------------------------------------------------

        for key in metrics.keys():
            if key not in self.fieldnames:
                self.fieldnames.append(key)

        row = {k: metrics.get(k, "") for k in self.fieldnames}

        self.writer.writerow(row)

        self._step += 1

        if self._step % self.flush_every == 0:
            self.file.flush()

    # --------------------------------------------------

    def flush(self):

        if self.enabled and self.file:
            self.file.flush()

    # --------------------------------------------------

    def close(self):

        if self.enabled and self.file:
            self.file.close()
            self.file = None

    # --------------------------------------------------

    def __del__(self):

        try:
            self.close()
        except Exception:
            pass