---
phase: 3
plan: 1
wave: 1
---

# Plan 3.1: Async RAFT Synthesis Engine

## Objective
Build the async GPT-4o synthesis engine that generates RAFT triplets concurrently. This is the core of the data pipeline — quality here directly determines fine-tuning quality.

## Context
- .gsd/SPEC.md
- .gsd/ARCHITECTURE.md
- .gsd/DECISIONS.md (ADR-003, ADR-004)
- src/config.py
- src/data/embedding.py

## Tasks

<task type="auto">
  <name>Write async RAFT synthesis module</name>
  <files>
    src/raft/synthesis.py
    src/raft/__init__.py
  </files>
  <action>
    Write src/raft/synthesis.py with:

    SYSTEM_PROMPT: mortgage domain expert, outputs grounded ONLY in provided chunk,
      never fabricate rates/percentages/eligibility criteria/financial figures.

    GENERATION_PROMPT: Given chunk, generate JSON with:
      - "question": realistic question a 35-65 year old mortgage seeker would ask
      - "chain_of_thought": step-by-step reasoning citing doc with ##begin_quote## / ##end_quote## markers
      - "answer": warm, jargon-light conversational English, 2-4 sentences

    get_async_openai_client() -> AsyncOpenAI

    _generate_triplet_async(client, semaphore, chunk_text, chunk_metadata, collection_name, num_distractors=3):
      - async with semaphore
      - Call GPT-4o with response_format=json_object
      - Parse JSON response
      - Run ChromaDB query in thread executor (sync, can't be awaited)
      - Filter oracle chunk from distractor results
      - Return dict: {question, oracle_doc, oracle_metadata, distractor_docs, chain_of_thought, answer}
      - Return None on any exception (do not crash the batch)

    generate_batch_async(chunks, collection_name, questions_per_chunk=5, output_path, concurrency=30, progress_callback=None):
      - Build task list: len(chunks) × questions_per_chunk coroutines
      - asyncio.as_completed for streaming writes as results arrive
      - Write each valid triplet immediately to output JSONL (don't buffer all in RAM)
      - Call progress_callback(completed, total) after each result
      - Return count of successfully written triplets

    generate_batch(chunks, ...): sync wrapper via asyncio.run()

    CRITICAL: questions_per_chunk=5 default (not 3). We need 20K from ~4K chunks.
    CRITICAL: concurrency=30 default. Do NOT go higher without confirming API tier.
    CRITICAL: write to file immediately in as_completed loop — 20K triplets won't fit in RAM cleanly.
  </action>
  <verify>python -c "from src.raft.synthesis import generate_batch; print('synthesis module OK')"</verify>
  <done>Imports without error</done>
</task>
