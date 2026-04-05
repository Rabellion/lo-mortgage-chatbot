"""
P2 (ML Lead) — Day 2 task.

Unsloth + TRL SFTTrainer fine-tuning pipeline for Gemma 4 31B.

All GPU/model imports are LAZY — this module is importable on CPU machines
(config editing, dataset inspection) without requiring torch or unsloth.

Usage:
    from src.training.finetune import run_finetuning
    from src.training.config import TrainingConfig

    cfg = TrainingConfig(run_name="gemma4-31b-raft-run1")
    output_dir = run_finetuning(cfg)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.config import TrainingConfig


def _load_sharegpt_dataset(dataset_path: str) -> list[dict]:
    """Load ShareGPT JSON array from disk."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run Phase 3 (RAFT synthesis) first to generate train_sharegpt.json"
        )
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return data


def _format_sharegpt(examples: dict, tokenizer) -> dict:
    """
    Convert ShareGPT conversation format to model input strings.

    ShareGPT schema:
        {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

    Remaps "human" → "user", "gpt" → "assistant" for apply_chat_template.
    """
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    texts = []

    for convs in examples["conversations"]:
        messages = [
            {"role": role_map.get(turn["from"], turn["from"]), "content": turn["value"]}
            for turn in convs
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    return {"text": texts}


def run_finetuning(config: "TrainingConfig | None" = None) -> str:
    """
    Run the full fine-tuning pipeline.

    Args:
        config: TrainingConfig instance. Defaults to TrainingConfig() with all defaults.

    Returns:
        Path to the merged 16-bit output directory.
    """
    # ── Lazy imports (GPU machine only) ───────────────────────────────────
    import torch
    import wandb
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    from src.training.config import TrainingConfig

    if config is None:
        config = TrainingConfig()

    print(config.summary())

    # ── WandB ─────────────────────────────────────────────────────────────
    wandb.init(
        project=config.wandb_project,
        name=config.run_name,
        config=config.model_dump(),
    )

    # ── Load model + tokenizer ────────────────────────────────────────────
    print(f"\nLoading {config.base_model} ...")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[config.dtype]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        dtype=torch_dtype,
        load_in_4bit=False,  # 16-bit LoRA — no QLoRA
    )

    # ── Apply LoRA ────────────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth long-context checkpointing
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Load + format dataset ─────────────────────────────────────────────
    print(f"\nLoading dataset from {config.dataset_path} ...")
    raw = _load_sharegpt_dataset(config.dataset_path)
    print(f"  {len(raw):,} examples loaded")

    hf_dataset = Dataset.from_list(raw)
    hf_dataset = hf_dataset.map(
        lambda examples: _format_sharegpt(examples, tokenizer),
        batched=True,
        remove_columns=hf_dataset.column_names,
        desc="Formatting ShareGPT → chat template",
    )

    split = hf_dataset.train_test_split(test_size=0.05, seed=config.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset):,}  |  Eval: {len(eval_dataset):,}")

    # ── SFT config ────────────────────────────────────────────────────────
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        optim=config.optim,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        bf16=(config.dtype == "bfloat16"),
        fp16=(config.dtype == "float16"),
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        seed=config.seed,
        report_to="wandb",
        run_name=config.run_name,
        packing=config.packing,
        dataset_num_proc=os.cpu_count(),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print("\nStarting training ...")
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU: {torch.cuda.get_device_name(0)}  ({gpu_mem_gb:.1f} GB)")

    trainer_stats = trainer.train()

    print(f"\nTraining complete.")
    print(f"  Runtime  : {trainer_stats.metrics.get('train_runtime', 0):.0f}s")
    print(f"  Steps    : {trainer_stats.metrics.get('train_steps_per_second', 0):.2f} steps/s")
    print(f"  Loss     : {trainer_stats.metrics.get('train_loss', 0):.4f}")

    # ── Save merged 16-bit model ──────────────────────────────────────────
    merged_dir = config.merged_output_dir()
    print(f"\nSaving merged 16-bit model to {merged_dir} ...")
    model.save_pretrained_merged(
        merged_dir,
        tokenizer,
        save_method="merged_16bit",
    )
    print("  Merged model saved.")

    # ── Optional HuggingFace Hub push ─────────────────────────────────────
    if config.hub_model_id:
        print(f"\nPushing to HuggingFace Hub: {config.hub_model_id} ...")
        model.push_to_hub_merged(
            config.hub_model_id,
            tokenizer,
            save_method="merged_16bit",
        )
        print("  Hub push complete.")

    wandb.finish()

    print(f"\nDone. Model at: {merged_dir}")
    return merged_dir
