"""
Runtime training execution.
"""

import torch
from contextlib import nullcontext

from stackformer.training.loops import train_loop, validation_loop
from stackformer.logging.metrics import MetricTracker
from stackformer.utils.utils import move_to_device, is_main_process


class Engine:
    """
    Core runtime executor for training and evaluation.

    Responsibilities
    ----------------
    • Execute training steps
    • Execute evaluation steps
    • Handle gradient accumulation
    • Apply optimizer / scheduler
    • Track metrics
    • Send metrics to logging backends
    """

    def __init__(
        self,
        state,
        grad_accum_steps=1,
        max_grad_norm=None,
        monitor=None,
        log_every=10,
    ):

        self.state = state

        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm

        self.monitor = monitor
        self.live_monitoring = monitor is not None

        self.log_every = log_every

        self.metrics = MetricTracker()

        self._accum_step = 0

        # Move model to device
        if self.state.model is not None:
            self.state.model.to(self.state.device)

    # -----------------------------------------------------

    def train_one_epoch(self, dataloader, epoch):

        self.metrics.reset()

        train_loop(self, dataloader, epoch)

    # -----------------------------------------------------

    def validate_one_epoch(self, dataloader, epoch):

        validation_loop(self, dataloader, epoch)

    # -----------------------------------------------------

    def _train_step(self, batch):

        state = self.state
        scaler = state.scaler

        self.metrics.start_step_timer()

        # -------------------------------------------------
        # Batch handling
        # -------------------------------------------------

        if isinstance(batch, (list, tuple)):
            inputs, targets = batch
        elif isinstance(batch, dict):
            inputs = batch["inputs"]
            targets = batch["targets"]
        else:
            raise ValueError("Batch must be tuple/list or dict")

        inputs = move_to_device(inputs, state.device)
        targets = move_to_device(targets, state.device)

        # -------------------------------------------------
        # AMP context
        # -------------------------------------------------

        if scaler and scaler.is_enabled:
            autocast_ctx = scaler.autocast()
        else:
            autocast_ctx = nullcontext()

        # -------------------------------------------------
        # Forward
        # -------------------------------------------------

        with autocast_ctx:

            outputs = state.model(inputs)

            loss = state.config["criterion"](
                outputs.view(-1, outputs.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        loss = loss / self.grad_accum_steps

        # -------------------------------------------------
        # Backward
        # -------------------------------------------------

        if scaler and scaler.is_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        self._accum_step += 1

        grad_norm = None

        # -------------------------------------------------
        # Optimizer step
        # -------------------------------------------------

        if self._accum_step % self.grad_accum_steps == 0:

            if self.max_grad_norm is not None:
                grad_norm = self._clip_gradients()

            self._optimizer_step()
            self._scheduler_step()

            state.optimizer.zero_grad(set_to_none=True)

            state.step()
            self._accum_step = 0

        # update batch index (checkpoint resume support)
        state.batch_idx += 1

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        step_time = self.metrics.end_step_timer()

        loss_value = loss.item() * self.grad_accum_steps

        self.metrics.update("loss", loss_value)

        lr = self.get_lr()

        if lr is not None:
            self.metrics.update("learning_rate", lr)

        if step_time is not None:
            self.metrics.update("step_time", step_time)

        # tokens/sec for LLM training
        if torch.is_tensor(inputs) and inputs.dim() >= 2:
            tokens = inputs.numel()
            self.metrics.update_tokens(tokens)

        self.metrics.update_perplexity(loss_value)

        if grad_norm is not None:
            self.metrics.update("grad_norm", grad_norm)

        # -------------------------------------------------
        # Logging
        # -------------------------------------------------

        metrics = self.metrics.get_all()
        metrics["lr"] = lr

        if state.global_step % self.log_every == 0:
            self._log_step(metrics)

        return metrics

    # -----------------------------------------------------

    def _eval_step(self, batch):

        state = self.state
        scaler = state.scaler

        if isinstance(batch, (list, tuple)):
            inputs, targets = batch
        elif isinstance(batch, dict):
            inputs = batch["inputs"]
            targets = batch["targets"]
        else:
            raise ValueError("Batch must be tuple/list or dict")

        inputs = move_to_device(inputs, state.device)
        targets = move_to_device(targets, state.device)

        if scaler and scaler.is_enabled:
            autocast_ctx = scaler.autocast()
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:

            outputs = state.model(inputs)

            loss = state.config["criterion"](
                outputs.view(-1, outputs.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        loss_value = loss.item()

        self.metrics.update("val_loss", loss_value)

        metrics = {"val_loss": loss_value}

        self._log_step(metrics)

        return metrics

    # -----------------------------------------------------

    def _optimizer_step(self):

        state = self.state
        scaler = state.scaler

        if scaler and scaler.is_enabled:
            scaler.step(state.optimizer)
            scaler.update()
        else:
            state.optimizer.step()

    # -----------------------------------------------------

    def _scheduler_step(self):

        state = self.state

        if state.scheduler is not None:
            state.scheduler.step()

    # -----------------------------------------------------

    def _clip_gradients(self):

        state = self.state
        scaler = state.scaler

        if scaler and scaler.is_enabled:
            scaler.unscale_(state.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            state.model.parameters(),
            self.max_grad_norm,
        )

        return float(grad_norm)

    # -----------------------------------------------------

    def _log_step(self, metrics):

        if self.monitor is None:
            return

        # Only rank 0 logs in distributed training
        if not is_main_process():
            return

        self.monitor.log(metrics)

    # -----------------------------------------------------

    def get_lr(self):

        opt = self.state.optimizer

        if opt is None:
            return None

        return opt.param_groups[0]["lr"]