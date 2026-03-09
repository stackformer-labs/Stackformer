"""
stackformer.logging.metrics

Advanced metric tracking utilities for research training loops.

Design goals
------------
• Extremely lightweight (runs every step)
• Streaming statistics (no history storage)
• Compatible with distributed training
• Clear API for researchers
• Works for LLM, vision, and segmentation tasks
"""

from collections import defaultdict
import math
import time


class _Metric:
    """
    Internal metric container.

    Maintains:
        sum
        count
        average
        last value
    """

    def __init__(self):
        self.reset()

    # --------------------------------------------------

    def update(self, value):

        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return

        self.last = value
        self.sum += value
        self.count += 1

    # --------------------------------------------------

    def avg(self):

        if self.count == 0:
            return 0.0

        return self.sum / self.count

    # --------------------------------------------------

    def reset(self):

        self.sum = 0.0
        self.count = 0
        self.last = None


# ======================================================
# Metric Tracker
# ======================================================


class MetricTracker:
    """
    Streaming metric tracker for training loops.

    Designed to run every step with minimal overhead.
    """

    def __init__(self):

        self.metrics = defaultdict(_Metric)

        self.start_time = time.time()
        self.step_start_time = None

    # --------------------------------------------------

    def update(self, name, value):
        """
        Update a metric value.
        """

        self.metrics[name].update(value)

    # --------------------------------------------------

    def avg(self, name):
        """
        Return running average.
        """

        metric = self.metrics.get(name)

        if metric is None:
            return None

        return metric.avg()

    # --------------------------------------------------

    def last(self, name):
        """
        Return last metric value.
        """

        metric = self.metrics.get(name)

        if metric is None:
            return None

        return metric.last

    # --------------------------------------------------

    def compute(self, name):
        """
        Compatibility alias.
        """

        return self.avg(name)

    # --------------------------------------------------

    def reset(self):
        """
        Reset all metrics.
        """

        for metric in self.metrics.values():
            metric.reset()

        self.start_time = time.time()

    # --------------------------------------------------

    def get_all(self):
        """
        Return all metric averages.
        """

        results = {}

        for name, metric in self.metrics.items():
            results[name] = metric.avg()

        return results

    # ==================================================
    # Timing utilities
    # ==================================================

    def start_step_timer(self):
        """
        Start step timing.
        """

        self.step_start_time = time.time()

    # --------------------------------------------------

    def end_step_timer(self):
        """
        End step timer and record step_time metric.
        """

        if self.step_start_time is None:
            return None

        step_time = time.time() - self.step_start_time

        self.update("step_time", step_time)

        self.step_start_time = None

        return step_time

    # ==================================================
    # Throughput metrics
    # ==================================================

    def update_tokens(self, tokens):
        """
        Track tokens/sec (LLM training).
        """

        self.update("tokens", tokens)

        total_tokens = self.metrics["tokens"].sum
        elapsed = time.time() - self.start_time

        if elapsed > 0:
            tokens_per_sec = total_tokens / elapsed
            self.metrics["tokens_per_sec"].last = tokens_per_sec

    # --------------------------------------------------

    def update_samples(self, samples):
        """
        Track samples/sec (vision tasks).
        """

        self.update("samples", samples)

        total_samples = self.metrics["samples"].sum
        elapsed = time.time() - self.start_time

        if elapsed > 0:
            samples_per_sec = total_samples / elapsed
            self.metrics["samples_per_sec"].last = samples_per_sec

    # ==================================================
    # Derived metrics
    # ==================================================

    def update_perplexity(self, loss):
        """
        Compute perplexity from cross-entropy loss.

        PPL = exp(loss)
        """

        try:
            ppl = math.exp(loss)
        except OverflowError:
            ppl = float("inf")

        self.update("perplexity", ppl)

    # ==================================================
    # Utilities
    # ==================================================

    def __contains__(self, name):

        return name in self.metrics

    # --------------------------------------------------

    def __repr__(self):

        metrics = self.get_all()

        parts = []

        for name, value in metrics.items():

            if isinstance(value, float):
                parts.append(f"{name}: {value:.4f}")
            else:
                parts.append(f"{name}: {value}")

        return " | ".join(parts)