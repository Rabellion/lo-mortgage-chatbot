# DECISIONS.md — Architecture Decision Record

## ADR-001: Gemma 4 31B Dense (NOT MoE 26B)
**Date**: 2026-04-04
**Status**: Decided
**Decision**: Use `google/gemma-4-31b-it` (dense) not the MoE 26B variant.
**Reason**: Dense model has more consistent instruction-following behavior for fine-tuning. MoE routing introduces variability that hurts RAFT training.

## ADR-002: 16-bit LoRA, No QLoRA
**Date**: 2026-04-04
**Status**: Decided
**Decision**: Train with full 16-bit LoRA, not QLoRA (4-bit quantized LoRA).
**Reason**: Dense 31B with QLoRA risks gradient degradation. H100 80GB has sufficient VRAM headroom for 16-bit LoRA with gradient checkpointing.

## ADR-003: GPT-4o for RAFT Synthesis
**Date**: 2026-04-04
**Status**: Decided
**Decision**: Use GPT-4o for triplet generation, not Gemini or Claude.
**Reason**: GPT-4o's structured JSON output mode (`response_format: json_object`) is reliable and well-tested for this pattern. Cost is acceptable.

## ADR-004: 30 Concurrent Workers for Synthesis
**Date**: 2026-04-04
**Status**: Decided
**Decision**: Default concurrency = 30 async workers via semaphore.
**Reason**: Tier-2 OpenAI limit is 5,000 RPM. At 30 workers × ~2s/call = ~900 RPM. Safe headroom. Adjustable via `--concurrency` flag.

## ADR-005: 20K Triplet Target (Generate 24K Raw)
**Date**: 2026-04-04
**Status**: Decided
**Decision**: Target 20K validated triplets, generate 24K raw (20% buffer).
**Reason**: Validation rejects ~15-20% of triplets (missing CoT citations, fabricated rates). Buffer avoids a second generation run just to fill quota.
