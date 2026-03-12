"""Unified logger interface for CSV/TensorBoard/W&B backends."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

from stackformer.logging.csv_logger import CSVLogger
from stackformer.logging.tensorboard_logger import TensorBoardLogger
from stackformer.utils.utils import print_once


class Logger:
    """Multiplex metrics to one or more logging backends."""

    def __init__(
        self,
        csv: bool = True,
        tensorboard: bool = False,
        wandb: bool = False,
        log_dir: str = "logs",
        experiment_name: str = "run",
        wandb_project: str = "stackformer",
        wandb_config: dict | None = None,
    ):
        self.backends: List[Any] = []
        self._failed_backends: set[str] = set()

        if csv:
            self.backends.append(CSVLogger(log_dir=log_dir, filename=f"{experiment_name}_metrics.csv"))
            print_once("[Logger] CSV logging enabled")

        if tensorboard:
            self.backends.append(TensorBoardLogger(log_dir=log_dir, experiment_name=experiment_name))
            print_once("[Logger] TensorBoard logging enabled")

        if wandb:
            from stackformer.logging.wandb_logger import WandBLogger

            self.backends.append(
                WandBLogger(project=wandb_project, experiment_name=experiment_name, config=wandb_config)
            )
            print_once("[Logger] Weights & Biases logging enabled")

        if not self.backends:
            print_once("[Logger] No logging backend enabled")

    def log(self, metrics: Dict[str, float]) -> None:
        if not metrics:
            return
        for backend in self.backends:
            backend_name = backend.__class__.__name__
            if backend_name in self._failed_backends:
                continue
            try:
                backend.log(metrics)
            except Exception as exc:
                if backend_name not in self._failed_backends:
                    self._failed_backends.add(backend_name)
                    warnings.warn(
                        f"Logger backend '{backend_name}' failed and will be skipped: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

    def close(self) -> None:
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

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
