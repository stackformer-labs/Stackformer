"""Backward-compatible Trainer import path.

This module is deprecated. Prefer `stackformer.engine.Trainer`.
Legacy implementation is available at `stackformer.legacy.trainer`.
"""

from __future__ import annotations

import warnings

from stackformer.engine.trainer import Trainer as _EngineTrainer

warnings.warn(
    "`stackformer.trainer.Trainer` is deprecated and now aliases `stackformer.engine.Trainer`. "
    "Use `from stackformer.engine import Trainer` for new code. "
    "Legacy trainer remains at `stackformer.legacy.trainer.Trainer`.",
    DeprecationWarning,
    stacklevel=2,
)

Trainer = _EngineTrainer

__all__ = ["Trainer"]
