"""Checkpoint save/load utilities.

Three complementary, purpose-built checkpoint mechanisms live here:

1. SafeTensors (``save`` / ``load``)
   Full, *consolidated* model weights. Always a single, unsharded
   state_dict written by the main process -- safe to publish, diff, or
   load into a plain ``nn.Module`` elsewhere. Works for plain, DDP-, or
   FSDP/FSDP2-wrapped models: sharded parameters are gathered into a
   full state_dict first (a collective op), so every rank must call
   ``save`` / ``load`` even though only rank 0 touches disk.

2. torch.save (also part of ``save`` / ``load``)
   The "private" resume state that rides alongside the weights: full
   optimizer state, scheduler state, AMP scaler state, epoch/step
   counters, seed, and the training config (lr, loss fn, ...). Not meant
   to be shared -- only to resume *this* run.

3. DCP -- torch.distributed.checkpoint (``save_sharded`` / ``load_sharded``)
   For models whose parameters are sharded across ranks (FSDP/FSDP2,
   tensor-parallel, ...). Every rank writes only its own shard; shards
   are *never* gathered or merged into one file. Restoring can happen at
   a different world size -- DCP reshards automatically. This is the
   cheap, scalable path for routine checkpointing of large models;
   reserve ``save``/``load`` for occasional, exportable snapshots.

Notes on torch version support: (1)/(2) are FSDP-aware and (3) works at
all only on torch>=2.2 (``torch.distributed.checkpoint.state_dict``). On
older torch, (1)/(2) silently fall back to plain ``.state_dict()`` (fine
for plain/DDP models, incorrect for FSDP) and (3) raises a clear error.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file, save_file

from stackformer.utils.utils import is_main_process, print_once

try:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
        get_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
        set_state_dict,
    )
    from torch.distributed.checkpoint.stateful import Stateful

    _DCP_AVAILABLE = True
except ImportError:  # torch < 2.2: no DCP state-dict helpers available
    _DCP_AVAILABLE = False

    class Stateful:  # type: ignore[no-redef]
        """No-op stand-in so this module still imports on old torch."""


class _AppState(Stateful):
    """Wraps model (+ optimizer) so DCP can save/load their *sharded*
    local state directly -- no gathering, no merging.

    Works uniformly whether ``model`` is plain, DDP-, or FSDP/FSDP2-
    wrapped: ``get_state_dict`` / ``set_state_dict`` resolve the correct
    (replicated vs. sharded/DTensor) representation on their own. This
    mirrors the pattern from PyTorch's own DCP tutorial.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> Dict[str, Any]:
        model_state, optim_state = get_state_dict(self.model, self.optimizer)
        return {"model": model_state, "optim": optim_state}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )


class CheckpointManager:
    """Manage model checkpoints."""

    def __init__(
        self,
        output_dir: str,
        device: str | torch.device = "cpu",
    ):
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Consolidated: SafeTensors (weights) + torch.save (everything else)
    def save(self, state: Dict[str, Any], name: str = "latest") -> Dict[str, str] | None:
        """
        Save a consolidated, shareable checkpoint: full model weights
        (SafeTensors) + full training state (torch.save).

        Collective when the model is sharded (FSDP/FSDP2): gathering a
        full state_dict needs every rank's participation, so call this
        on *every* rank. Only the main process writes to disk.

        Returns
        -------
        dict
            {"weights": "...safetensors", "state": "...pt"}
            None on non-main ranks.
        """
        model = state.get("model")
        if model is None:
            raise ValueError("state['model'] is required.")

        model_state, optimizer_state = self._gather_full_state(model, state.get("optimizer"))

        if not is_main_process():
            return None

        weights_path = self._weights_path(name)
        state_path = self._state_path(name)

        self._safe_save_safetensors(model_state, weights_path)

        training_state = self._build_train_state(state, optimizer_state=optimizer_state)
        self._safe_torch_save(training_state, state_path)

        print_once(f"[Checkpoint] Weights saved -> {weights_path}")
        print_once(f"[Checkpoint] State saved   -> {state_path}")

        return {
            "weights": str(weights_path),
            "state": str(state_path),
        }

    def load(
        self,
        name: str,
        state: Dict[str, Any],
        broadcast_from_rank0: bool = False,
    ) -> Dict[str, Any]:
        """
        Restore model + training state saved by ``save``.

        Parameters
        ----------
        name
            Checkpoint name (e.g. "latest", "best")
        broadcast_from_rank0
            If True (needs torch>=2.2 and a live process group), only
            rank 0 reads the SafeTensors weights file and DCP reshards
            it out to every other rank -- use for large FSDP models on
            shared storage to avoid every rank paying the read cost.
            Default False: every rank reads its own copy, which is
            simpler and needs no process group.
        """
        weights_path = self._weights_path(name)
        state_path = self._state_path(name)

        if not weights_path.exists():
            raise FileNotFoundError(weights_path)
        if not state_path.exists():
            raise FileNotFoundError(state_path)

        model = state.get("model")
        if model is None:
            raise ValueError("state['model'] is required.")

        broadcast = broadcast_from_rank0 and _DCP_AVAILABLE
        should_read_weights = (not broadcast) or is_main_process()
        model_state = load_file(str(weights_path), device="cpu") if should_read_weights else {}

        self._scatter_full_model_state(model, model_state, broadcast_from_rank0=broadcast)

        checkpoint = torch.load(
            state_path,
            map_location=self.device,
            weights_only=True,
        )

        self._scatter_optimizer_state(model, state.get("optimizer"), checkpoint.get("optimizer_state_dict"))
        self._maybe_load_state_dict(state.get("scheduler"), checkpoint.get("scheduler_state_dict"), "scheduler")
        self._maybe_load_state_dict(state.get("scaler"), checkpoint.get("scaler_state_dict"), "scaler")

        print_once(f"[Checkpoint] Loaded <- {name}")

        return {
            "epoch": checkpoint.get("epoch", 0),
            "global_step": checkpoint.get("global_step", 0),
            "batch_idx": checkpoint.get("batch_idx", 0),
            "config": checkpoint.get("config", {}),
            "seed": checkpoint.get("seed"),
            "jit_model_path": checkpoint.get("jit_model_path"),
        }

    # Sharded: DCP -- shards written independently, never merged
    def save_sharded(self, state: Dict[str, Any], name: str = "latest") -> Dict[str, str]:
        """
        Save each rank's local shard of the model (+ optimizer) via DCP.
        Shards are never gathered into one file -- cheap and scalable
        for large FSDP/FSDP2 models.

        Collective: every rank must call this (it writes its own shard).
        Small, rank-identical bookkeeping (epoch/step/config/seed/
        scheduler/scaler) is still written once, by the main process, as
        a plain torch.save file next to the DCP shards.
        """
        self._require_dcp()

        model = state.get("model")
        if model is None:
            raise ValueError("state['model'] is required.")

        # Do NOT unwrap DDP/FSDP here -- get_state_dict needs the wrapped
        # module to know how the parameters are sharded/replicated.
        app_state = _AppState(model, state.get("optimizer"))

        ckpt_dir = self._sharded_dir(name)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        dcp.save({"app": app_state}, checkpoint_id=str(ckpt_dir))

        result = {"sharded_dir": str(ckpt_dir)}

        if is_main_process():
            meta_path = self._sharded_meta_path(name)
            training_state = self._build_train_state(state, include_optimizer=False)
            self._safe_torch_save(training_state, meta_path)
            result["state"] = str(meta_path)
            print_once(f"[Checkpoint] Sharded weights -> {ckpt_dir}")
            print_once(f"[Checkpoint] State saved      -> {meta_path}")

        return result

    def load_sharded(self, name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Restore a checkpoint written by ``save_sharded``.

        Collective: every rank must call this. ``state['model']`` (and
        optimizer, if resuming it) must already be constructed with
        whatever sharding layout *this* run uses -- it need not match
        the world size that wrote the checkpoint; DCP reshards
        automatically.
        """
        self._require_dcp()

        ckpt_dir = self._sharded_dir(name)
        meta_path = self._sharded_meta_path(name)

        if not ckpt_dir.exists():
            raise FileNotFoundError(ckpt_dir)
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)

        model = state.get("model")
        if model is None:
            raise ValueError("state['model'] is required.")

        app_state = _AppState(model, state.get("optimizer"))
        dcp.load({"app": app_state}, checkpoint_id=str(ckpt_dir))

        checkpoint = torch.load(
            meta_path,
            map_location=self.device,
            weights_only=True,
        )

        self._maybe_load_state_dict(state.get("scheduler"), checkpoint.get("scheduler_state_dict"), "scheduler")
        self._maybe_load_state_dict(state.get("scaler"), checkpoint.get("scaler_state_dict"), "scaler")

        print_once(f"[Checkpoint] Loaded (sharded) <- {name}")

        return {
            "epoch": checkpoint.get("epoch", 0),
            "global_step": checkpoint.get("global_step", 0),
            "batch_idx": checkpoint.get("batch_idx", 0),
            "config": checkpoint.get("config", {}),
            "seed": checkpoint.get("seed"),
            "jit_model_path": checkpoint.get("jit_model_path"),
        }

    # TorchScript
    def save_jit_model( self, model: torch.nn.Module, name: str = "latest",) -> str:

        if not is_main_process():
            return ""

        model = self._unwrap_model(model)
        model.eval()

        path = self._jit_path(name)
        tmp = path.with_suffix(".tmp")

        try:
            scripted = torch.jit.script(model)
            torch.jit.save(scripted, tmp)
            os.replace(tmp, path)

            print_once(f"[Checkpoint] TorchScript -> {path}")
            return str(path)

        except Exception as exc:
            if tmp.exists():
                tmp.unlink()

            raise RuntimeError(
                f"TorchScript save failed: {exc}"
            ) from exc

    def load_jit_model( self, path: str,) -> torch.jit.ScriptModule:

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        model = torch.jit.load(
            str(path),
            map_location=self.device,
        )

        model.eval()
        return model

    # Paths
    def _weights_path(self, name: str) -> Path:
        return self.output_dir / f"checkpoint_{name}.safetensors"

    def _state_path(self, name: str) -> Path:
        return self.output_dir / f"checkpoint_{name}.pt"

    def _jit_path(self, name: str) -> Path:
        return self.output_dir / f"checkpoint_{name}.jit.pt"

    def _sharded_dir(self, name: str) -> Path:
        return self.output_dir / f"checkpoint_{name}_dcp"

    def _sharded_meta_path(self, name: str) -> Path:
        return self._sharded_dir(name) / "train_state.pt"

    # Full-state gather/scatter helpers (SafeTensors + torch.save path)
    @staticmethod
    def _gather_full_state(
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
    ):
        """
        Collective-safe gather of a full (unsharded) model/optimizer
        state dict, regardless of wrapping (plain / DDP / FSDP/FSDP2).

        Must be called on every rank when the model may be sharded: DCP's
        ``full_state_dict=True`` all-gathers shards onto rank 0 and
        returns an empty dict everywhere else, so this is safe to gate
        on ``is_main_process()`` immediately afterwards.
        """
        if _DCP_AVAILABLE:
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)

            model_state = get_model_state_dict(model=model, options=options)
            model_state = {
                k: (v.contiguous() if isinstance(v, torch.Tensor) else v)
                for k, v in model_state.items()
            }

            optimizer_state = (
                get_optimizer_state_dict(model=model, optimizers=optimizer, options=options)
                if optimizer is not None
                else None
            )
            return model_state, optimizer_state

        # Fallback for torch < 2.2: no sharded-state helpers, so `model`
        # is assumed plain or DDP-wrapped (state_dict() is already the
        # full, replicated state).
        unwrapped = CheckpointManager._unwrap_model(model)
        model_state = {
            k: v.detach().cpu().contiguous() for k, v in unwrapped.state_dict().items()
        }
        optimizer_state = CheckpointManager._safe_state_dict(optimizer)
        return model_state, optimizer_state

    def _scatter_full_model_state(
        self,
        model: torch.nn.Module,
        model_state: Dict[str, Any],
        broadcast_from_rank0: bool = False,
    ) -> None:
        if _DCP_AVAILABLE:
            options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=broadcast_from_rank0)
            set_model_state_dict(model=model, model_state_dict=model_state, options=options)
            return

        unwrapped = self._unwrap_model(model)
        unwrapped.load_state_dict(model_state)
        unwrapped.to(self.device)

    def _scatter_optimizer_state(
        self,
        model: torch.nn.Module,
        optimizer: Any,
        optimizer_state: Optional[Dict[str, Any]],
    ) -> None:
        if optimizer is None or optimizer_state is None:
            return

        if _DCP_AVAILABLE:
            try:
                options = StateDictOptions(full_state_dict=True)
                set_optimizer_state_dict(
                    model=model,
                    optimizers=optimizer,
                    optim_state_dict=optimizer_state,
                    options=options,
                )
            except Exception as exc:
                print_once(f"[Checkpoint] Warning: failed to restore optimizer: {exc}")
            return

        self._maybe_load_state_dict(optimizer, optimizer_state, "optimizer")

    def _build_train_state(
        self,
        state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]] = None,
        include_optimizer: bool = True,
    ) -> Dict[str, Any]:
        train_state = {
            "scheduler_state_dict": self._safe_state_dict(state.get("scheduler")),
            "scaler_state_dict": self._safe_state_dict(state.get("scaler")),
            "epoch": state.get("epoch", 0),
            "global_step": state.get("global_step", 0),
            "batch_idx": state.get("batch_idx", 0),
            "config": state.get("config", {}),
            "seed": state.get("seed", 42),
            "jit_model_path": state.get("jit_model_path"),
        }
        if include_optimizer:
            train_state["optimizer_state_dict"] = optimizer_state
        return train_state

    @staticmethod
    def _require_dcp() -> None:
        if not _DCP_AVAILABLE:
            raise RuntimeError(
                "Sharded (DCP) checkpointing needs torch>=2.2 "
                "(torch.distributed.checkpoint.state_dict). Detected "
                f"torch=={torch.__version__}."
            )

    # Helpers
    @staticmethod
    def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        return model.module if hasattr(model, "module") else model

    @staticmethod
    def _safe_state_dict(obj: Any) -> Optional[Dict[str, Any]]:
        if obj is None or not hasattr(obj, "state_dict"):
            return None

        try:
            return obj.state_dict()
        except Exception:
            return None

    @staticmethod
    def _maybe_load_state_dict(
        obj: Any,
        state_dict: Optional[Dict[str, Any]],
        label: str,
    ):

        if obj is None or state_dict is None:
            return

        if not hasattr(obj, "load_state_dict"):
            return

        try:
            obj.load_state_dict(state_dict)

        except Exception as exc:
            print_once(
                f"[Checkpoint] Warning: failed to restore {label}: {exc}"
            )

    @staticmethod
    def _safe_torch_save(
        obj: Dict[str, Any],
        path: Path,
    ):

        tmp = path.with_suffix(path.suffix + ".tmp")

        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)

        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    @staticmethod
    def _safe_save_safetensors(
        tensors: Dict[str, torch.Tensor],
        path: Path,
    ):

        tmp = path.with_suffix(path.suffix + ".tmp")

        try:
            save_file(tensors, str(tmp))
            os.replace(tmp, path)

        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)