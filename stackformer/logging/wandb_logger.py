"""Weights & Biases logger backend."""

from __future__ import annotations

import time
from typing import Dict, Optional

from stackformer.utils.utils import is_main_process


class WandBLogger:
    """Write scalar metrics to Weights & Biases."""

    def __init__(
        self,
        project: str = "stackformer",
        experiment_name: Optional[str] = None,
        config: Optional[dict] = None,
        entity: Optional[str] = None,
        watch_model: bool = False,
    ):
        self.enabled = is_main_process()
        self.run = None
        self.step = 0
        self.start_time = time.time()
        self.watch_model_enabled = watch_model

        if not self.enabled:
            return

        try:
            import wandb
        except ImportError as exc:
            raise ImportError("wandb is not installed. Install with `pip install wandb`.") from exc

        self.wandb = wandb
        self.run = self.wandb.init(project=project, name=experiment_name, entity=entity, config=config)

    def log(self, metrics: Dict[str, float]) -> None:
        if not self.enabled or not metrics:
            return

        clean_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        clean_metrics["step"] = self.step
        clean_metrics["time"] = time.time() - self.start_time

        try:
            self.wandb.log(clean_metrics, step=self.step)
        except Exception:
            pass

        self.step += 1

    def log_model(self, model) -> None:
        if not self.enabled or not self.watch_model_enabled:
            return
        try:
            self.wandb.watch(model)
        except Exception:
            pass

    def finish(self) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            self.wandb.finish()
        except Exception:
            pass

    def __del__(self):
        try:
            self.finish()
        except Exception:
            pass
