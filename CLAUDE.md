# LO Mortgage Chatbot — Project Context

## What this is
RAFT fine-tuned mortgage chatbot. Gemma 4 31B Dense, trained on 20K RAFT triplets.
Target: <3% hallucination, >0.85 RAGAS faithfulness, >4.2/5 tone score.

## Stack
- Model: google/gemma-4-31b-it (Apache 2.0)
- Fine-tuning: Unsloth + TRL SFTTrainer, 16-bit LoRA (no QLoRA)
- Vector store: Pinecone
- Synthesis: AsyncOpenAI GPT-4o, 30 concurrent workers
- Inference: vLLM + FastAPI + SSE streaming
- Hardware: 1x H100 80GB on RunPod

## Team roles
- P1: Data Lead — ingestion, chunking, Pinecone
- P2 (Huzaifa): ML Lead — RAFT synthesis, fine-tuning, evaluation
- P3: RAG Engineer — retriever, inference engine, FastAPI
- P4: Frontend — React + Vite + Tailwind
- P5: Eval/QA — RAGAS, DeepEval, tone rubric

## Current phase
Phase 1 complete (scaffold, config, CFPB loader, download script).
Phase 2 (ingestion + chunking + Pinecone) — P1 working on it.
Phase 3 (RAFT synthesis) — P2 next after Phase 2 done.

## Key numbers
- Target triplets: 20K (generate 24K raw for 20% buffer)
- Chunk size: 2000 chars (~512 tokens)
- Top-K retrieval: 5
- LoRA r=16, alpha=32, lr=2e-4, 3 epochs

## Planning docs
- .gsd/SPEC.md — full spec and success criteria
- .gsd/ROADMAP.md — 6 phases overview
- .gsd/phases/ — atomic task plans per phase
- .gsd/STATE.md — current blockers and status

## Data sources
- CFPB complaints (training) — data/raw/cfpb_complaints.json
- FinanceBench (eval only, never training) — data/raw/financebench.jsonl
- Manual PDFs: CFPB handbook, FHA/VA guides (data/raw/)

## Important rules
- FinanceBench is ALWAYS eval_only=True — never chunk or embed into training store
- Generate 24K triplets to guarantee 20K after validation filtering
- 16-bit LoRA only — no QLoRA for dense 31B
