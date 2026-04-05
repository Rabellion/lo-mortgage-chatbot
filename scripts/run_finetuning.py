"""
P2 (ML Lead) — Day 3 task.

Entry point for launching a fine-tuning run on the H100.

Usage:
    python scripts/run_finetuning.py
    python scripts/run_finetuning.py --run-name gemma4-31b-raft-run2 --epochs 3
    python scripts/run_finetuning.py --smoke-test          # 50-step smoke test
    python scripts/run_finetuning.py --output-dir models/run2

Validates dataset and GPU memory before touching the model —
fail fast, waste no GPU time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root without editable install
sys.path.insert(0, str(Path(__file__).parent.parent))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Gemma 4 31B RAFT fine-tuning")
    p.add_argument("--run-name", default=None, help="WandB run name (overrides config default)")
    p.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    p.add_argument("--output-dir", default=None, help="Output directory for checkpoints + merged model")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 50 steps only — validates the pipeline without full GPU cost",
    )
    p.add_argument(
        "--dataset-path",
        default=None,
        help="Path to train_sharegpt.json (overrides config default)",
    )
    return p.parse_args()


def _check_dataset(path: Path, min_examples: int = 1000) -> int:
    """Validate dataset exists and has enough examples. Returns count."""
    if not path.exists():
        print(f"FAIL  Dataset not found: {path}")
        print("      Run Phase 3 (RAFT synthesis) first to generate train_sharegpt.json")
        sys.exit(1)

    try:
        with path.open() as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"FAIL  Dataset JSON parse error: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        print(f"FAIL  Expected JSON array, got {type(data).__name__}")
        sys.exit(1)

    n = len(data)
    if n < min_examples:
        print(f"FAIL  Only {n} examples in dataset — need at least {min_examples}")
        print("      Wait for P1 to generate more triplets before launching training.")
        sys.exit(1)

    # Spot-check structure
    sample = data[0]
    if "conversations" not in sample:
        print(f"FAIL  Dataset items must have a 'conversations' key (ShareGPT format)")
        print(f"      First item keys: {list(sample.keys())}")
        sys.exit(1)

    return n


def _check_gpu() -> None:
    """Verify CUDA is available and print GPU info. Exits if no GPU found."""
    try:
        import torch
    except ImportError:
        print("FAIL  torch not installed — run pip install torch")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("FAIL  No CUDA GPU detected — this script must run on the H100")
        sys.exit(1)

    props = torch.cuda.get_device_properties(0)
    gb = props.total_memory / 1024**3
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3

    print(f"OK    GPU: {props.name}")
    print(f"OK    Total VRAM : {gb:.1f} GB")
    print(f"OK    Free VRAM  : {free_gb:.1f} GB")

    if gb < 70:
        print(f"WARN  GPU has only {gb:.1f} GB — Gemma 4 31B in 16-bit needs ~70 GB")
        print("      Consider reducing max_seq_length or using a larger GPU.")


def main() -> None:
    args = _parse_args()

    # ── Imports (config only — no GPU libs yet) ───────────────────────────
    from src.training.config import TrainingConfig

    cfg = TrainingConfig()

    # Apply CLI overrides
    if args.run_name:
        cfg = cfg.model_copy(update={"run_name": args.run_name})
    if args.epochs:
        cfg = cfg.model_copy(update={"num_train_epochs": args.epochs})
    if args.output_dir:
        cfg = cfg.model_copy(update={"output_dir": args.output_dir})
    if args.dataset_path:
        cfg = cfg.model_copy(update={"dataset_path": args.dataset_path})
    if args.smoke_test:
        cfg = cfg.model_copy(update={"max_steps": 50, "run_name": cfg.run_name + "-smoke"})
        print("WARN  Smoke test mode: running only 50 steps")

    # ── Pre-flight checks ─────────────────────────────────────────────────
    print("\n── Dataset validation ──────────────────────────────────────")
    dataset_path = Path(cfg.dataset_path)
    n = _check_dataset(dataset_path, min_examples=1 if args.smoke_test else 1000)
    print(f"OK    Dataset: {n:,} examples in {dataset_path}")

    print("\n── GPU check ───────────────────────────────────────────────")
    _check_gpu()

    print("\n── Config ──────────────────────────────────────────────────")
    print(cfg.summary())

    if not args.smoke_test:
        print("\nAll checks passed. Starting training in 3 seconds...")
        print("Monitor at: https://wandb.ai (project: lo-mortgage-raft)")
        import time
        time.sleep(3)

    # ── Run ───────────────────────────────────────────────────────────────
    from src.training.finetune import run_finetuning
    output = run_finetuning(cfg)
    print(f"\nRun complete. Merged model: {output}")


if __name__ == "__main__":
    main()
