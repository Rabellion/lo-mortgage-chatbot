"""
P2 (ML Lead) — Day 5 task.

Evaluate a saved checkpoint or merged model against a set of mortgage questions.
Produces a JSON results file for comparison across runs.

Usage:
    python scripts/eval_checkpoint.py --model-dir models/gemma-4-31b-raft/merged
    python scripts/eval_checkpoint.py --model-dir models/gemma-4-31b-raft/checkpoint-500
    python scripts/eval_checkpoint.py --model-dir models/run2/merged --output data/evaluation/run2.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# 20 mortgage questions covering: rates, eligibility, process, refinancing, documents
EVAL_QUESTIONS = [
    "What credit score do I need to qualify for a conventional mortgage?",
    "How much down payment is required for a first-time home buyer?",
    "What is the difference between a fixed-rate and adjustable-rate mortgage?",
    "Can I refinance my mortgage if my home is underwater?",
    "What documents do I need to apply for a mortgage?",
    "What is PMI and when can I remove it?",
    "How does debt-to-income ratio affect mortgage approval?",
    "What is the maximum loan amount for a conforming conventional loan?",
    "How long does the mortgage underwriting process typically take?",
    "What is the difference between pre-qualification and pre-approval?",
    "Can self-employed borrowers get a mortgage?",
    "What is an FHA loan and who qualifies for it?",
    "How does a home appraisal affect mortgage approval?",
    "What are closing costs and how much should I expect to pay?",
    "What is a rate lock and how long does it last?",
    "Can I use gift funds for a down payment?",
    "What happens if I miss a mortgage payment?",
    "What is the difference between a home equity loan and a HELOC?",
    "How does bankruptcy affect my ability to get a mortgage?",
    "What is mortgage insurance and when is it required?",
]

SYSTEM_PROMPT = (
    "You are an expert mortgage advisor. Answer the question accurately based on "
    "standard US mortgage guidelines. If you cite a specific rate or figure, "
    "state the source (e.g., 'per CFPB guidelines' or 'per Fannie Mae SEL'). "
    "Be concise but complete."
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a fine-tuned checkpoint")
    p.add_argument("--model-dir", required=True, help="Path to merged model or checkpoint dir")
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: data/evaluation/<model_dir_name>.json)",
    )
    p.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens per response")
    p.add_argument("--questions", default=None, help="Path to custom questions JSON array (optional)")
    return p.parse_args()


def _load_model(model_dir: str):
    """Load model for inference using Unsloth FastLanguageModel."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("FAIL  unsloth not installed — run pip install unsloth")
        sys.exit(1)

    import torch

    print(f"Loading model from {model_dir} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _generate(model, tokenizer, question: str, max_new_tokens: int) -> tuple[str, float]:
    """Run inference for a single question. Returns (response_text, latency_seconds)."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    # Note: must pass text= as keyword — Unsloth's Gemma 4 processor patch
    # maps the first positional arg to images, not text.
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(text=prompt, return_tensors="pt").to("cuda")["input_ids"]

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - t0

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    response = tokenizer.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response, latency


def _check_hallucination_signals(response: str) -> list[str]:
    """
    Lightweight heuristic: flag responses that state specific rates/percentages
    without any hedging or source attribution.
    Not a replacement for full RAGAS eval — just a quick signal.
    """
    import re

    flags = []
    # Unhedged specific rate mention (e.g., "the rate is 6.5%")
    rate_pattern = re.compile(r"\b(?:rate|APR|yield)\s+(?:is|was|will be)\s+\d+\.?\d*\s*%", re.I)
    if rate_pattern.search(response):
        flags.append("unhedged_rate_claim")

    # Specific dollar thresholds stated as fact without attribution
    dollar_pattern = re.compile(r"\$\d[\d,]+\b")
    source_words = re.compile(r"\b(?:per|according|source|fannie|freddie|cfpb|hud|fha|guideline)\b", re.I)
    if dollar_pattern.search(response) and not source_words.search(response):
        flags.append("unattributed_dollar_figure")

    return flags


def main() -> None:
    args = _parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"FAIL  Model directory not found: {model_dir}")
        sys.exit(1)

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("data/evaluation") / f"{model_dir.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Questions
    if args.questions:
        with open(args.questions) as f:
            questions = json.load(f)
    else:
        questions = EVAL_QUESTIONS

    model, tokenizer = _load_model(str(model_dir))

    results = []
    total_latency = 0.0
    hallucination_count = 0

    print(f"\nEvaluating {len(questions)} questions ...\n")
    for i, q in enumerate(questions, 1):
        response, latency = _generate(model, tokenizer, q, args.max_new_tokens)
        flags = _check_hallucination_signals(response)
        if flags:
            hallucination_count += 1

        total_latency += latency
        print(f"[{i:02d}/{len(questions)}] ({latency:.1f}s) {'[FLAGGED] ' if flags else ''}{q[:60]}...")

        results.append({
            "question": q,
            "response": response,
            "latency_s": round(latency, 2),
            "hallucination_flags": flags,
        })

    # Summary
    avg_latency = total_latency / len(questions)
    flag_rate = hallucination_count / len(questions)

    summary = {
        "model_dir": str(model_dir),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "num_questions": len(questions),
        "avg_latency_s": round(avg_latency, 2),
        "flagged_count": hallucination_count,
        "flag_rate": round(flag_rate, 4),
        "results": results,
    }

    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n── Summary ─────────────────────────────────────────────────")
    print(f"  Questions evaluated : {len(questions)}")
    print(f"  Avg latency         : {avg_latency:.2f}s")
    print(f"  Flagged responses   : {hallucination_count} ({flag_rate*100:.1f}%)")
    print(f"  Results saved to    : {out_path}")
    print("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
