"""Runtime step execution for training and evaluation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, Optional, Tuple

import torch

from stackformer.logging.metrics import MetricTracker
from stackformer.training.loops import train_loop, validation_loop
from stackformer.utils.utils import is_main_process, move_to_device


class Engine:
    """Executes train/eval steps with robust batch and loss handling."""

    def __init__(
        self,
        state,
        grad_accum_steps: int = 1,
        max_grad_norm: Optional[float] = None,
        monitor: Optional[Any] = None,
        log_every: int = 10,
        max_train_steps: Optional[int] = None,
        batch_parser: Optional[Callable[[Any], Tuple[Any, Any]]] = None,
        compute_loss_fn: Optional[Callable[[Any, Any, Any], torch.Tensor]] = None,
    ):
        self.state = state
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.max_grad_norm = max_grad_norm
        self.max_train_steps = max_train_steps
        self.monitor = monitor
        self.live_monitoring = monitor is not None
        self.log_every = max(1, int(log_every))
        self.batch_parser = batch_parser
        self.compute_loss_fn = compute_loss_fn
        self.metrics = MetricTracker()
        self._accum_step = 0

        if self.state.model is not None:
            self.state.model.to(self.state.device)

    def train_one_epoch(self, dataloader, epoch: int) -> None:
        self.metrics.reset()
        train_loop(self, dataloader, epoch)

    def validate_one_epoch(self, dataloader, epoch: int) -> None:
        validation_loop(self, dataloader, epoch)

    def _train_step(self, batch) -> dict:
        state = self.state
        if state.optimizer is None:
            raise ValueError("Optimizer is required for training steps.")

        self.metrics.start_step_timer()
        inputs, targets = self._prepare_batch(batch)
        scaler = state.scaler

        with self._get_autocast_context(scaler):
            outputs = self._forward_model(inputs)
            loss = self._compute_loss(outputs, targets)

        loss = loss / self.grad_accum_steps

        if scaler and scaler.is_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        self._accum_step += 1
        grad_norm = None

        if self._accum_step % self.grad_accum_steps == 0:
            if self.max_grad_norm is not None:
                grad_norm = self._clip_gradients()
            self._optimizer_step()
            self._scheduler_step()
            state.optimizer.zero_grad(set_to_none=True)
            state.step()
            self._accum_step = 0

        state.batch_idx += 1

        step_time = self.metrics.end_step_timer()
        loss_value = float(loss.item() * self.grad_accum_steps)
        self.metrics.update("loss", loss_value)

        lr = self.get_lr()
        if lr is not None:
            self.metrics.update("learning_rate", lr)
        if step_time is not None:
            self.metrics.update("step_time", step_time)
        if torch.is_tensor(inputs) and inputs.dim() >= 2:
            self.metrics.update_tokens(inputs.numel())

        self.metrics.update_perplexity(loss_value)
        if grad_norm is not None:
            self.metrics.update("grad_norm", grad_norm)

        metrics = self.metrics.get_all()
        metrics["lr"] = lr
        if state.global_step % self.log_every == 0:
            self._log_step(metrics)
        return metrics

    def _eval_step(self, batch) -> dict:
        inputs, targets = self._prepare_batch(batch)
        scaler = self.state.scaler

        with self._get_autocast_context(scaler):
            outputs = self._forward_model(inputs)
            loss = self._compute_loss(outputs, targets)

        loss_value = float(loss.item())
        self.metrics.update("val_loss", loss_value)
        metrics = {"val_loss": loss_value}
        self._log_step(metrics)
        return metrics

    def _optimizer_step(self) -> None:
        state = self.state
        scaler = state.scaler
        if scaler and scaler.is_enabled:
            scaler.step(state.optimizer)
            scaler.update()
        else:
            state.optimizer.step()

    def _scheduler_step(self) -> None:
        scheduler = self.state.scheduler
        if scheduler is None:
            return
        if scheduler.__class__.__name__.lower() == "reducelronplateau":
            return
        scheduler.step()

    def _clip_gradients(self) -> float:
        state = self.state
        scaler = state.scaler
        if scaler and scaler.is_enabled:
            scaler.unscale_(state.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), self.max_grad_norm)
        return float(grad_norm)

    def _log_step(self, metrics: dict) -> None:
        if self.monitor is None or not is_main_process():
            return
        self.monitor.log(metrics)

    def get_lr(self) -> Optional[float]:
        opt = self.state.optimizer
        if opt is None:
            return None
        return float(opt.param_groups[0]["lr"])

    def reached_max_train_steps(self) -> bool:
        if self.max_train_steps is None:
            return False
        return self.state.global_step >= int(self.max_train_steps)

    def _prepare_batch(self, batch):
        if self.batch_parser is not None:
            inputs, targets = self.batch_parser(batch)
            return move_to_device(inputs, self.state.device), move_to_device(targets, self.state.device)

        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
            return move_to_device(inputs, self.state.device), move_to_device(targets, self.state.device)

        if isinstance(batch, dict):
            targets = batch.get("targets", batch.get("labels"))
            if targets is None:
                raise ValueError("Dictionary batch must include `targets` or `labels`.")
            model_inputs = {k: v for k, v in batch.items() if k not in {"targets", "labels"}}
            if "inputs" in model_inputs and len(model_inputs) == 1:
                model_inputs = model_inputs["inputs"]
            return move_to_device(model_inputs, self.state.device), move_to_device(targets, self.state.device)

        raise ValueError("Unsupported batch format. Use (inputs, targets) or dict with labels/targets.")

    def _forward_model(self, inputs):
        if isinstance(inputs, dict):
            return self.state.model(**inputs)
        return self.state.model(inputs)

    def _compute_loss(self, outputs, targets) -> torch.Tensor:
        if self.compute_loss_fn is not None:
            return self.compute_loss_fn(outputs, targets, self.state)

        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss

        if isinstance(outputs, (list, tuple)) and outputs:
            if torch.is_tensor(outputs[0]) and outputs[0].ndim == 0:
                return outputs[0]
            outputs = outputs[0]

        criterion = self.state.config.get("criterion")
        if criterion is None:
            raise ValueError("No criterion found. Pass criterion to Trainer or compute_loss_fn to Engine.")

        try:
            return criterion(outputs, targets)
        except TypeError:
            return criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=-100)

    @staticmethod
    def _get_autocast_context(scaler):
        if scaler and getattr(scaler, "is_enabled", False) and hasattr(scaler, "autocast"):
            return scaler.autocast()
        return nullcontext()
