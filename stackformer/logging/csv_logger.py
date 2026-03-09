"""CSV logger backend."""

from __future__ import annotations

import csv
import os
from typing import Dict

from stackformer.utils.utils import is_main_process


class CSVLogger:
    """Write scalar metrics to CSV.

    Args:
        log_dir: Output log directory.
        filename: CSV filename.
    """

    def __init__(self, log_dir: str = "logs", filename: str = "metrics.csv"):
        self.enabled = is_main_process()
        self.file = None
        self.writer = None
        self.headers_written = False

        if not self.enabled:
            return

        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        self.file = open(self.path, mode="a", newline="", encoding="utf-8")

    def log(self, metrics: Dict[str, float]) -> None:
        if not self.enabled or not metrics:
            return
        if self.file is None:
            return

        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(metrics.keys()))

        if not self.headers_written:
            self.writer.writeheader()
            self.headers_written = True

        clean = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        self.writer.writerow(clean)

    def flush(self) -> None:
        if self.enabled and self.file:
            self.file.flush()

    def close(self) -> None:
        if self.enabled and self.file:
            self.file.flush()
            self.file.close()
            self.file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
