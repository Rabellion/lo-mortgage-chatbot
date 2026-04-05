# SPEC.md — Project Specification

> **Status**: `FINALIZED`

## Vision

A RAFT-based conversational AI mortgage advisor for clients aged 35–65. The system fine-tunes Google Gemma 4 31B on 20K+ synthetically generated mortgage Q&A triplets grounded in authoritative source documents (CFPB, Fannie Mae, HUD). Every answer must cite a source document — hallucination is a legal liability, not just a quality issue.

## Goals

1. **Grounded Knowledge** — All answers traceable to a source document chunk; hallucinated rates/figures are a hard failure
2. **20K RAFT Triplets** — Sufficient training data to meaningfully shift model behavior, generated concurrently via GPT-4o
3. **Fine-tuned Gemma 4 31B** — LoRA-adapted model that outperforms baseline RAG on mortgage domain Q&A
4. **Measurable Quality** — Hallucination rate < 3%, RAGAS faithfulness > 0.85, tone score > 4.2/5
5. **Production-ready Inference API** — FastAPI + vLLM serving the fine-tuned model with RAG at inference time

## Non-Goals (Out of Scope)

- CRM / HubSpot integration (post-MVP)
- Multi-language support
- Real-time rate feeds (all rates come from static documents)
- QLoRA — 16-bit LoRA only for dense 31B model
- Frontend UI (API only for MVP)

## Users

- **End users**: Clients 35–65 seeking mortgage/loan guidance
- **ML owner**: Huzaifa Imran — owns full ML pipeline (data → training → evaluation)
- **Infra**: API team handles FastAPI + Docker + vLLM serving

## Constraints

- Hardware: 1× H100 80GB (RunPod / Lambda Labs) — provision early
- Model: `google/gemma-4-31b-it` (Apache 2.0) — NOT the MoE 26B variant
- Timeline: 10-day MVP sprint (started 2026-04-04)
- Fine-tuning: Run #1 on ~20K triplets; Run #2 optional on 8–12K additional if metrics miss
- OpenAI API: GPT-4o for RAFT synthesis — ~30 concurrent workers safe under tier-2 limits
- Source documents: CFPB (API), FinanceBench (GitHub), Fannie Mae / HUD / Freddie Mac (manual PDFs)

## Success Criteria

- [ ] ChromaDB vector store built from ≥ 3 authoritative mortgage document sources
- [ ] ≥ 20,000 validated RAFT triplets in ShareGPT format
- [ ] Gemma 4 31B fine-tuned run completes without OOM on H100 80GB
- [ ] Hallucination rate < 3% on held-out eval set
- [ ] RAGAS faithfulness score > 0.85
- [ ] Tone score > 4.2 / 5.0
- [ ] Inference API responds correctly to 5 sample mortgage questions

---

*Last updated: 2026-04-04*
