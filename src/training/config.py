"""
P2 (ML Lead) — Day 2 task.

Locked training hyperparameters for Gemma 4 31B LoRA fine-tuning.
Uses Pydantic BaseModel (not BaseSettings) — these are code constants,
not env-driven. Override via TrainingConfig(run_name="run2", ...) at call site.

IMPORTANT: Verify these before spinning up the H100 — wasted GPU time = real money.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class TrainingConfig(BaseModel):
    # ── Model ─────────────────────────────────────────────────────────────
    base_model: str = "google/gemma-4-31b-it"
    dataset_path: str = "data/raft_triplets/train_sharegpt.json"
    output_dir: str = "models/gemma-4-31b-raft"
    hub_model_id: str = ""  # empty = skip HuggingFace push

    # ── LoRA ──────────────────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # ── Sequence ──────────────────────────────────────────────────────────
    max_seq_length: int = 4096
    dtype: str = "bfloat16"  # H100 native; do NOT change to fp16

    # ── Batch / gradient ─────────────────────────────────────────────────
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # effective batch size = 8
    gradient_checkpointing: bool = True

    # ── Optimiser ─────────────────────────────────────────────────────────
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"  # Unsloth fused 8-bit Adam

    # ── Epochs / steps ────────────────────────────────────────────────────
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 = run full epochs; set > 0 to override for smoke tests

    # ── Checkpointing / logging ──────────────────────────────────────────
    save_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 100
    eval_strategy: str = "steps"
    save_total_limit: int = 3  # keep only last 3 checkpoints on disk

    # ── WandB ─────────────────────────────────────────────────────────────
    wandb_project: str = "lo-mortgage-raft"
    run_name: str = "gemma4-31b-raft-run1"

    # ── Misc ──────────────────────────────────────────────────────────────
    seed: int = 42
    packing: bool = False  # SFTTrainer sequence packing — off for RAFT format

    @field_validator("dtype")
    @classmethod
    def _dtype_valid(cls, v: str) -> str:
        allowed = {"bfloat16", "float16", "float32"}
        if v not in allowed:
            raise ValueError(f"dtype must be one of {allowed}")
        return v

    def merged_output_dir(self) -> str:
        return f"{self.output_dir}/merged"

    def summary(self) -> str:
        lines = [
            "── Training config ──────────────────────────────────────────",
            f"  base_model              : {self.base_model}",
            f"  dataset_path            : {self.dataset_path}",
            f"  output_dir              : {self.output_dir}",
            f"  hub_model_id            : {self.hub_model_id or '(not pushing)'}",
            "",
            f"  lora_r / lora_alpha     : {self.lora_r} / {self.lora_alpha}",
            f"  lora_dropout            : {self.lora_dropout}",
            f"  target_modules          : {', '.join(self.target_modules)}",
            "",
            f"  max_seq_length          : {self.max_seq_length}",
            f"  dtype                   : {self.dtype}",
            f"  batch_size (per device) : {self.per_device_train_batch_size}",
            f"  gradient_accum_steps    : {self.gradient_accumulation_steps}",
            f"  effective batch size    : {self.per_device_train_batch_size * self.gradient_accumulation_steps}",
            "",
            f"  learning_rate           : {self.learning_rate}",
            f"  lr_scheduler            : {self.lr_scheduler_type}",
            f"  warmup_ratio            : {self.warmup_ratio}",
            f"  num_train_epochs        : {self.num_train_epochs}",
            f"  max_steps               : {'full epochs' if self.max_steps == -1 else self.max_steps}",
            "",
            f"  save_steps              : {self.save_steps}",
            f"  logging_steps           : {self.logging_steps}",
            f"  wandb_project           : {self.wandb_project}",
            f"  run_name                : {self.run_name}",
            "─────────────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)
