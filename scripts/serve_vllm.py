"""
P2 (ML Lead) — Day 5/6 task.

Launch the vLLM OpenAI-compatible server for the fine-tuned Gemma 4 31B model.
This bridges P2 (training) → P3 (RAG API): set RAG_LLM_PROVIDER=vllm and
VLLM_BASE_URL=http://localhost:8000/v1 in the RAG repo's .env.

Usage:
    python scripts/serve_vllm.py --model-dir models/gemma-4-31b-raft/merged
    python scripts/serve_vllm.py --model-dir models/gemma-4-31b-raft/merged --port 8001
    python scripts/serve_vllm.py --model-dir models/run2/merged --quantize int8

This script calls `vllm serve` as a subprocess with recommended settings for
Gemma 4 31B on a single H100 80GB. Adjust --gpu-memory-utilization if OOM.

After launching, update the RAG repo .env:
    RAG_LLM_PROVIDER=vllm
    RAG_LLM_MODEL=<model-dir-name>
    VLLM_BASE_URL=http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve fine-tuned model with vLLM")
    p.add_argument("--model-dir", required=True, help="Path to merged model directory")
    p.add_argument("--port", type=int, default=8000, help="Port for vLLM server (default: 8000)")
    p.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.92,
        help="Fraction of GPU memory for vLLM KV cache (default: 0.92)",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096, matches training config)",
    )
    p.add_argument(
        "--quantize",
        default=None,
        choices=["int8", "fp8", "awq", "gptq"],
        help="Optional quantization (int8 recommended if VRAM is tight after loading)",
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1 for single H100)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"FAIL  Model directory not found: {model_dir}")
        print("      Run scripts/run_finetuning.py first to produce the merged model.")
        sys.exit(1)

    # Model name for the OpenAI-compatible API endpoint
    # RAG repo must set RAG_LLM_MODEL to this value
    model_name = model_dir.name

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_dir),
        "--served-model-name", model_name,
        "--host", args.host,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--trust-remote-code",
        "--enable-prefix-caching",
    ]

    if args.quantize:
        cmd += ["--quantization", args.quantize]

    print("── vLLM serve ───────────────────────────────────────────────")
    print(f"  Model dir       : {model_dir}")
    print(f"  Served as       : {model_name}")
    print(f"  Endpoint        : http://{args.host}:{args.port}/v1")
    print(f"  GPU mem util    : {args.gpu_memory_utilization}")
    print(f"  Max seq len     : {args.max_model_len}")
    print(f"  Quantization    : {args.quantize or 'none (16-bit)'}")
    print("─────────────────────────────────────────────────────────────")
    print()
    print("RAG repo .env settings once server is up:")
    print(f"  RAG_LLM_PROVIDER=vllm")
    print(f"  RAG_LLM_MODEL={model_name}")
    print(f"  VLLM_BASE_URL=http://localhost:{args.port}/v1")
    print()
    print(f"Running: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nvLLM server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\nFAIL  vLLM exited with code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
