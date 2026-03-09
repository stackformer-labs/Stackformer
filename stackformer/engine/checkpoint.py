"""Checkpoint save/load utilities.

This module provides robust checkpointing for training state and optional
TorchScript model export.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from stackformer.utils.utils import is_main_process, print_once


class CheckpointManager:
    """Manage training checkpoints.

    Args:
        output_dir: Directory to store checkpoint files.
        device: Device used for loading checkpoints.
    """

    def __init__(self, output_dir: str, device: str | torch.device = "cpu"):
        self.output_dir = output_dir
        self.device = device
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, state: Dict[str, Any], name: str = "latest") -> str | None:
        """Save model and optimizer state.

        Args:
            state: Runtime state dictionary with keys like ``model``, ``optimizer``.
            name: Checkpoint suffix name.

        Returns:
            Saved path, or ``None`` for non-main distributed ranks.
        """
        if not is_main_process():
            return None

        model = state.get("model")
        if model is None:
            raise ValueError("`state['model']` is required to save a checkpoint.")

        model = self._unwrap_model(model)
        path = self._get_checkpoint_path(name)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self._safe_state_dict(state.get("optimizer")),
            "scheduler_state_dict": self._safe_state_dict(state.get("scheduler")),
            "scaler_state_dict": self._safe_state_dict(state.get("scaler")),
            "epoch": state.get("epoch", 0),
            "global_step": state.get("global_step", 0),
            "batch_idx": state.get("batch_idx", 0),
            "config": state.get("config", {}),
            "jit_model_path": state.get("jit_model_path"),
        }

        self._safe_save(checkpoint, path)
        print_once(f"[Checkpoint] Saved -> {path}")
        return path

    def load(self, path: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Load checkpoint and restore runtime state.

        Args:
            path: Checkpoint path.
            state: Runtime state dictionary.

        Returns:
            Metadata dict (epoch/global_step/batch_idx/config/jit_model_path).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        model = state.get("model")
        if model is None:
            raise ValueError("`state['model']` is required to load a checkpoint.")

        model = self._unwrap_model(model)
        model.load_state_dict(checkpoint["model_state_dict"])

        self._maybe_load_state_dict(state.get("optimizer"), checkpoint.get("optimizer_state_dict"), "optimizer")
        self._maybe_load_state_dict(state.get("scheduler"), checkpoint.get("scheduler_state_dict"), "scheduler")
        self._maybe_load_state_dict(state.get("scaler"), checkpoint.get("scaler_state_dict"), "scaler")

        metadata = {
            "epoch": checkpoint.get("epoch", 0),
            "global_step": checkpoint.get("global_step", 0),
            "batch_idx": checkpoint.get("batch_idx", 0),
            "config": checkpoint.get("config", {}),
            "jit_model_path": checkpoint.get("jit_model_path"),
        }
        print_once(f"[Checkpoint] Loaded <- {path}")
        return metadata

    def save_jit_model(self, model: torch.nn.Module, name: str = "latest_jit") -> str:
        """Save a scripted TorchScript model artifact.

        Args:
            model: PyTorch model to export.
            name: Artifact suffix.

        Returns:
            File path to scripted model.
        """
        if not is_main_process():
            return ""

        model = self._unwrap_model(model)
        model.eval()
        path = os.path.join(self.output_dir, f"{name}.jit.pt")
        tmp_path = f"{path}.tmp"

        try:
            scripted = torch.jit.script(model)
            torch.jit.save(scripted, tmp_path)
            os.replace(tmp_path, path)
            print_once(f"[Checkpoint] TorchScript saved -> {path}")
            return path
        except Exception as exc:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(f"TorchScript save failed: {exc}") from exc

    def load_jit_model(self, path: str) -> torch.jit.ScriptModule:
        """Load a TorchScript model artifact."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"TorchScript artifact not found: {path}")

        try:
            model = torch.jit.load(path, map_location=self.device)
            model.eval()
            return model
        except Exception as exc:
            raise RuntimeError(f"TorchScript load failed: {exc}") from exc

    def _get_checkpoint_path(self, name: str) -> str:
        return os.path.join(self.output_dir, f"checkpoint_{name}.pt")

    def _safe_state_dict(self, obj: Any) -> Optional[Dict[str, Any]]:
        if obj is None or not hasattr(obj, "state_dict"):
            return None
        try:
            return obj.state_dict()
        except Exception:
            return None

    def _maybe_load_state_dict(self, obj: Any, state_dict: Optional[Dict[str, Any]], label: str) -> None:
        if obj is None or state_dict is None:
            return
        if not hasattr(obj, "load_state_dict"):
            return
        try:
            obj.load_state_dict(state_dict)
        except Exception as exc:
            print_once(f"[Checkpoint] Warning: failed to restore {label}: {exc}")

    def _safe_save(self, obj: Dict[str, Any], path: str) -> None:
        tmp_path = f"{path}.tmp"
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
        except Exception as exc:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(f"Checkpoint save failed: {exc}") from exc

    @staticmethod
    def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        return model.module if hasattr(model, "module") else model
