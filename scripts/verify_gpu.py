"""
P2 (ML Lead) — Day 1 task.

GPU + Unsloth environment verification.
Run this first on the RunPod H100 before anything else.

Usage:
    python scripts/verify_gpu.py

Expected output on a healthy H100:
    ✓ CUDA available  — 1 device(s)
    ✓ Device 0: NVIDIA H100 80GB HBM3  (80.0 GB)
    ✓ bf16 supported
    ✓ torch version: 2.x.x
    ✓ unsloth importable
    ✓ transformers importable
    ✓ trl importable
    ✓ FastLanguageModel smoke test passed
    All checks passed — environment ready for Day 2.
"""

import sys


def check_cuda() -> str:
    import torch

    if not torch.cuda.is_available():
        return "FAIL  CUDA not available — check driver / CUDA install"

    n = torch.cuda.device_count()
    lines = [f"OK    CUDA available — {n} device(s)"]
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        gb = props.total_memory / 1024**3
        lines.append(f"OK    Device {i}: {props.name}  ({gb:.1f} GB)")

    # H100 supports bf16 natively
    if torch.cuda.is_bf16_supported():
        lines.append("OK    bf16 supported")
    else:
        lines.append("WARN  bf16 not supported — training will fall back to fp16")

    lines.append(f"OK    torch version: {torch.__version__}")
    return "\n".join(lines)


def check_imports() -> str:
    results = []
    for pkg in ("unsloth", "transformers", "trl", "peft", "wandb"):
        try:
            __import__(pkg)
            results.append(f"OK    {pkg} importable")
        except ImportError:
            results.append(f"FAIL  {pkg} not installed — run: pip install {pkg}")
    return "\n".join(results)


def check_unsloth_model() -> str:
    """
    Smoke-test FastLanguageModel without actually downloading Gemma 4 31B.
    Just verifies the Unsloth API surface is intact.
    """
    try:
        from unsloth import FastLanguageModel  # noqa: F401
        return "OK    FastLanguageModel importable (full model load skipped — use verify_gpu.py --full to pull weights)"
    except Exception as e:
        return f"FAIL  FastLanguageModel import error: {e}"


def main() -> None:
    full = "--full" in sys.argv  # reserved: actually load the model weights

    sections = [
        ("CUDA + hardware", check_cuda),
        ("Python packages", check_imports),
        ("Unsloth API", check_unsloth_model),
    ]

    all_ok = True
    for title, fn in sections:
        print(f"\n── {title} {'─' * (40 - len(title))}")
        output = fn()
        print(output)
        if "FAIL" in output:
            all_ok = False

    print()
    if all_ok:
        print("All checks passed — environment ready for Day 2.")
    else:
        print("One or more checks failed — fix before proceeding.")
        sys.exit(1)

    if full:
        print("\n── Full model load (--full) ─────────────────")
        print("Downloading google/gemma-4-31b-it weights via Unsloth ...")
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="google/gemma-4-31b-it",
                max_seq_length=512,
                load_in_4bit=False,
            )
            # One generation pass to confirm weights loaded correctly
            import torch
            FastLanguageModel.for_inference(model)
            messages = [{"role": "user", "content": "What is a mortgage?"}]
            # Note: must pass text= as keyword — Unsloth's Gemma 4 processor patch
            # maps the first positional arg to images, not text.
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            print(f"OK    Forward pass succeeded — weights valid")
            del model, tokenizer
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"FAIL  Model load error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
