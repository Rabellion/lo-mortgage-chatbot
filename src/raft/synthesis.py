"""
Async RAFT triplet synthesis engine.

Generates (question, oracle_doc, distractors, chain_of_thought, answer) triplets
via concurrent GPT-4o calls. Writes results to JSONL as they arrive — 20K triplets
won't fit cleanly in RAM.

Usage:
    from src.raft.synthesis import generate_batch

    count = generate_batch(
        chunks=chunks,
        output_path=Path("data/raft_triplets/raw_triplets.jsonl"),
    )
"""

import asyncio
import json
import random
from pathlib import Path
from typing import Any, Callable

from openai import AsyncOpenAI

from src.config import settings

SYSTEM_PROMPT = """You are a senior mortgage advisor and domain expert.
Your job is to generate realistic, high-quality training examples for a mortgage chatbot.

CRITICAL RULES:
- All facts, rates, percentages, and figures in your answer MUST come verbatim from the provided document.
- NEVER fabricate interest rates, credit score thresholds, down payment percentages, loan limits, or any financial figure.
- If the document does not contain a specific figure, do not include it.
- Write answers in warm, jargon-light conversational English suitable for a 35-65 year old home buyer.
- Chain-of-thought reasoning MUST use ##begin_quote## and ##end_quote## markers around every direct quote.
"""

GENERATION_PROMPT = """Based on the mortgage document below, generate ONE high-quality training example.

DOCUMENT:
{chunk_text}

Generate a JSON object (no markdown, just raw JSON) with exactly these fields:
{{
  "question": "A realistic question a mortgage seeker (age 35-65) would ask about this topic. At least 10 words.",
  "chain_of_thought": "Step-by-step reasoning. Quote the document with ##begin_quote## ... ##end_quote## markers. At least one quote required.",
  "answer": "Warm, conversational answer. 2-4 sentences. Only state figures that appear verbatim in the document above."
}}
"""


def get_async_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key)


async def _generate_triplet_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    chunk: dict[str, Any],
    all_chunks: list[dict[str, Any]],
    num_distractors: int = 3,
) -> dict[str, Any] | None:
    """Generate one RAFT triplet for a chunk. Returns None on failure."""
    chunk_text = chunk["text"]
    chunk_meta = chunk.get("metadata", {})

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=settings.synthesis_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": GENERATION_PROMPT.format(chunk_text=chunk_text),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=1024,
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)

            question = parsed.get("question", "").strip()
            cot = parsed.get("chain_of_thought", "").strip()
            answer = parsed.get("answer", "").strip()

            if not question or not cot or not answer:
                return None

            # Sample distractors (random chunks that aren't the oracle)
            pool = [c for c in all_chunks if c["text"] != chunk_text]
            distractors = random.sample(pool, min(num_distractors, len(pool)))

            return {
                "question": question,
                "oracle_doc": chunk_text,
                "oracle_metadata": chunk_meta,
                "distractor_docs": [d["text"] for d in distractors],
                "chain_of_thought": cot,
                "answer": answer,
            }

        except Exception:
            return None


async def _generate_batch_async(
    chunks: list[dict[str, Any]],
    output_path: Path,
    questions_per_chunk: int = 5,
    concurrency: int = 30,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """Async core of generate_batch. Streams results to JSONL as they arrive."""
    client = get_async_openai_client()
    semaphore = asyncio.Semaphore(concurrency)

    # Build task list: each chunk × questions_per_chunk
    tasks = [
        _generate_triplet_async(client, semaphore, chunk, chunks)
        for chunk in chunks
        for _ in range(questions_per_chunk)
    ]
    total = len(tasks)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    completed = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1

            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                written += 1

            if progress_callback:
                progress_callback(completed, total)

    return written


def generate_batch(
    chunks: list[dict[str, Any]],
    output_path: str | Path,
    questions_per_chunk: int = 5,
    concurrency: int = 30,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """
    Synchronous wrapper: generate RAFT triplets for all chunks.

    Writes results to output_path (JSONL, appended).
    Returns count of successfully written triplets.
    """
    output_path = Path(output_path)
    return asyncio.run(
        _generate_batch_async(
            chunks=chunks,
            output_path=output_path,
            questions_per_chunk=questions_per_chunk,
            concurrency=concurrency,
            progress_callback=progress_callback,
        )
    )
