"""Metric utilities and streaming tracker."""

from __future__ import annotations

from collections import defaultdict
import math
import time

import torch
import torch.distributed as dist


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def distributed_mean(value: float | torch.Tensor) -> float:
    tensor = torch.as_tensor(float(value), dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    pred_ids = preds.argmax(dim=-1) if preds.ndim > targets.ndim else preds
    return float((pred_ids == targets).float().mean().item())


def perplexity(loss: float | torch.Tensor) -> float:
    value = float(loss.item() if torch.is_tensor(loss) else loss)
    try:
        return float(math.exp(value))
    except OverflowError:
        return float("inf")


def precision_recall_f1(preds: torch.Tensor, targets: torch.Tensor):
    pred_ids = preds.argmax(dim=-1) if preds.ndim > targets.ndim else preds
    pred_ids = pred_ids.view(-1)
    targets = targets.view(-1)

    tp = ((pred_ids == 1) & (targets == 1)).sum().item()
    fp = ((pred_ids == 1) & (targets == 0)).sum().item()
    fn = ((pred_ids == 0) & (targets == 1)).sum().item()

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return float(precision), float(recall), float(f1)


def precision(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return precision_recall_f1(preds, targets)[0]


def recall(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return precision_recall_f1(preds, targets)[1]


def f1_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return precision_recall_f1(preds, targets)[2]


class _Metric:
    def __init__(self):
        self.reset()

    def update(self, value):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return
        self.last = value
        self.sum += value
        self.count += 1

    def avg(self):
        return 0.0 if self.count == 0 else self.sum / self.count

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.last = None


class MetricTracker:
    def __init__(self):
        self.metrics = defaultdict(_Metric)
        self.start_time = time.time()
        self.step_start_time = None

    def update(self, name, value):
        self.metrics[name].update(value)

    def avg(self, name):
        metric = self.metrics.get(name)
        return None if metric is None else metric.avg()

    def compute(self, name):
        return self.avg(name)

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()
        self.start_time = time.time()

    def get_all(self, reduce_distributed: bool = False):
        results = {name: metric.avg() for name, metric in self.metrics.items()}
        if reduce_distributed:
            return {name: distributed_mean(value) for name, value in results.items()}
        return results

    def start_step_timer(self):
        self.step_start_time = time.time()

    def end_step_timer(self):
        if self.step_start_time is None:
            return None
        step_time = time.time() - self.step_start_time
        self.update("step_time", step_time)
        self.step_start_time = None
        return step_time

    def update_tokens(self, tokens):
        self.update("tokens", tokens)
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            tokens_per_sec = self.metrics["tokens"].sum / elapsed
            self.update("tokens_per_sec", tokens_per_sec)   

    def update_perplexity(self, loss):
        self.update("perplexity", perplexity(loss))
