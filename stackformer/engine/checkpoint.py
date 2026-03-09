import os
import torch

from stackformer.utils.utils import print_once, is_main_process


class CheckpointManager:
    """
    Handles saving and loading training checkpoints.

    Features
    --------
    • Safe checkpoint saving (atomic write)
    • Resume training support
    • Optional optimizer / scheduler / scaler restore
    • Device-safe loading
    • DDP-safe checkpointing
    """

    def __init__(self, output_dir: str, device="cpu"):

        self.output_dir = output_dir
        self.device = device

        os.makedirs(self.output_dir, exist_ok=True)

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def save(self, state: dict, name: str = "latest"):
        """
        Save training state.
        """

        # Only main process saves checkpoint
        if not is_main_process():
            return

        path = self._get_checkpoint_path(name)

        model = state["model"]

        # unwrap DDP model if necessary
        if hasattr(model, "module"):
            model = model.module

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self._safe_state_dict(state.get("optimizer")),
            "scheduler_state_dict": self._safe_state_dict(state.get("scheduler")),
            "scaler_state_dict": self._safe_state_dict(state.get("scaler")),
            "epoch": state.get("epoch", 0),
            "global_step": state.get("global_step", 0),
            "batch_idx": state.get("batch_idx", 0),
            "config": state.get("config", {}),
        }

        self._safe_save(checkpoint, path)

        print_once(f"[Checkpoint] Saved → {path}")

    # -----------------------------------------------------

    def load(self, path: str, state: dict):
        """
        Load checkpoint and restore training state.
        """

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        model = state["model"]

        # unwrap DDP if needed
        if hasattr(model, "module"):
            model = model.module

        # -------------------------------------------------
        # Restore model
        # -------------------------------------------------

        model.load_state_dict(checkpoint["model_state_dict"])

        # -------------------------------------------------
        # Restore optimizer
        # -------------------------------------------------

        optimizer = state.get("optimizer")
        if optimizer is not None:
            opt_state = checkpoint.get("optimizer_state_dict")
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)

        # -------------------------------------------------
        # Restore scheduler
        # -------------------------------------------------

        scheduler = state.get("scheduler")
        if scheduler is not None:
            sch_state = checkpoint.get("scheduler_state_dict")
            if sch_state is not None:
                scheduler.load_state_dict(sch_state)

        # -------------------------------------------------
        # Restore AMP scaler
        # -------------------------------------------------

        scaler = state.get("scaler")
        if scaler is not None:
            scaler_state = checkpoint.get("scaler_state_dict")
            if scaler_state is not None:
                scaler.load_state_dict(scaler_state)

        # -------------------------------------------------
        # Metadata
        # -------------------------------------------------

        metadata = {
            "epoch": checkpoint.get("epoch", 0),
            "global_step": checkpoint.get("global_step", 0),
            "batch_idx": checkpoint.get("batch_idx", 0),
            "config": checkpoint.get("config", {}),
        }

        print_once(f"[Checkpoint] Loaded ← {path}")

        return metadata

    # -----------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------

    def _get_checkpoint_path(self, name: str):

        filename = f"checkpoint_{name}.pt"
        return os.path.join(self.output_dir, filename)

    # -----------------------------------------------------

    def _safe_state_dict(self, obj):
        """
        Safely get state_dict from optional objects.
        """

        if obj is None:
            return None

        return obj.state_dict()

    # -----------------------------------------------------

    def _safe_save(self, obj: dict, path: str):
        """
        Prevent corrupted checkpoints by saving to temp file first.
        """

        tmp_path = path + ".tmp"

        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)

        except Exception as e:

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            raise RuntimeError(f"Checkpoint save failed: {e}")