# STATE.md — Project Memory

## Current Position

- **Phase**: 1 — Data Acquisition
- **Task**: Not started
- **Status**: Planning complete, ready for execution

## Context

- Project initialized: 2026-04-04
- Sprint deadline: 2026-04-14 (10 days)
- ML owner: Huzaifa Imran
- H100 provisioned: ❌ Not yet — must be done before Phase 4

## Decisions Made

- Model: `google/gemma-4-31b-it` (dense, NOT MoE 26B)
- Fine-tuning: 16-bit LoRA via Unsloth + TRL (no QLoRA)
- Vector DB: ChromaDB persistent
- Synthesis: GPT-4o, 30 concurrent workers
- Target triplets: 20K validated (generate 24K raw to absorb ~15-20% validation loss)

## Blockers

- `data/raw/` is empty — Phase 1 must complete before anything else runs
- H100 not yet provisioned — needed for Phase 4

## Next Steps

1. `/execute 1` — Start Phase 1: Data Acquisition
