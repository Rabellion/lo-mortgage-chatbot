---
phase: 4
plan: 1
wave: 1
---

# Plan 4.1: Training Config + Unsloth Fine-tuning Pipeline

## Objective
Build the complete fine-tuning pipeline for Gemma 4 31B using Unsloth + TRL SFTTrainer. This runs on H100 80GB (RunPod/Lambda). Config must be locked before provisioning the GPU — wasted GPU time is wasted money.

## Context
- .gsd/SPEC.md
- .gsd/ARCHITECTURE.md
- .gsd/DECISIONS.md (ADR-001, ADR-002)
- data/raft_triplets/train_sharegpt.json (must exist — run Phase 3 first)

## Tasks

<task type="checkpoint:decision">
  <name>Lock training hyperparameters</name>
  <files>
    src/training/config.py
    src/training/__init__.py
  </files>
  <action>
    Write src/training/config.py as a Pydantic BaseModel (not BaseSettings — these are
    locked training params, not env-driven) with these defaults:

    base_model: str = "google/gemma-4-31b-it"
    dataset_path: str = "data/raft_triplets/train_sharegpt.json"
    output_dir: str = "models/gemma-4-31b-raft"
    hub_model_id: str = ""  # empty = don't push

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

    # Training
    max_seq_length: int = 4096
    dtype: str = "bfloat16"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # effective batch = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 = run full epochs
    save_steps: int = 100
    logging_steps: int = 10

    # WandB
    wandb_project: str = "lo-mortgage-raft"
    run_name: str = "gemma4-31b-raft-run1"

    PAUSE: Show user the config and ask for confirmation before continuing.
    Reason: Incorrect hyperparams = wasted H100 time = real money.
  </action>
  <verify>python -c "from src.training.config import TrainingConfig; c = TrainingConfig(); assert c.lora_r == 16 and c.per_device_train_batch_size == 1; print('Config OK')"</verify>
  <done>User confirms hyperparameters; config imports without error</done>
</task>

<task type="auto">
  <name>Write fine-tuning pipeline</name>
  <files>
    src/training/finetune.py
    scripts/run_finetuning.py
  </files>
  <action>
    Write src/training/finetune.py with run_finetuning(config=None):
    - Lazy imports (unsloth/trl/torch only on GPU machine)
    - wandb.init(project, name)
    - FastLanguageModel.from_pretrained(load_in_4bit=False) — 16-bit, NOT QLoRA
    - FastLanguageModel.get_peft_model() with config.target_modules
    - Load dataset from config.dataset_path as JSON array
    - format_sharegpt(): apply_chat_template converting "human"→"user", "gpt"→"assistant"
    - train_test_split(test_size=0.05, seed=42)
    - SFTConfig + SFTTrainer: bf16=True, eval_strategy="steps", report_to="wandb"
    - trainer.train()
    - model.save_pretrained_merged(output_dir+"/merged", save_method="merged_16bit")
    - If hub_model_id set: model.push_to_hub_merged()
    - wandb.finish()
    - Return output_dir

    Write scripts/run_finetuning.py:
    - Accepts --run-name, --epochs, --output-dir overrides
    - Validates data/raft_triplets/train_sharegpt.json exists and has >= 1000 examples
      (fail fast before spinning up GPU)
    - Prints GPU memory check: torch.cuda.get_device_properties(0).total_memory
    - Calls run_finetuning()

    DO NOT call any training imports at module level — lazy imports only.
    This file must be importable on CPU machines for testing.
  </action>
  <verify>python -c "from src.training.finetune import run_finetuning; print('finetune module importable OK')"</verify>
  <done>Imports without error on CPU (lazy imports mean torch/unsloth not required at import time)</done>
</task>
