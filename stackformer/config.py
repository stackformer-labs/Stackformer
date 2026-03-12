from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int
    embed_dim: int
    num_layers: int
    num_heads: int
    seq_len: int
    hidden_dim: int
    dropout: float = 0.0


@dataclass(slots=True)
class TrainingConfig:
    max_epochs: int = 1
    max_train_steps: int | None = None
    max_eval_steps: int | None = None
    eval_every_n_epochs: int = 1
    save_every_n_epochs: int = 1
    grad_accumulation_step: int = 1
    max_grad_norm: float | None = None
    lr: float = 3e-4
    weight_decay: float = 0.01
    optimizer_name: str = "adamw"
    scheduler_name: str = "none"
    warmup_steps: int = 0
    total_steps: int | None = None


@dataclass(slots=True)
class GenerationConfig:
    max_context_len: int = 128
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    eos_token_id: int | None = None
