# TODO.md

## Phase 1 — Data Acquisition
- [ ] Run Plan 1.1: scaffold + download CFPB + FinanceBench
- [ ] Manually add Fannie Mae Selling Guide PDF to data/raw/
- [ ] Manually add Freddie Mac Single-Family Guide PDF to data/raw/
- [ ] Manually add HUD FHA Handbook 4000.1 PDF to data/raw/
- [ ] Run Plan 1.2: CFPB JSON custom loader

## Phase 2 — Baseline RAG
- [ ] Run Plan 2.1: ingestion + chunking
- [ ] Run Plan 2.2: ChromaDB build + baseline retrieval check

## Phase 3 — RAFT Synthesis
- [ ] Run Plan 3.1: async synthesis engine
- [ ] Run Plan 3.2: validation + ShareGPT conversion
- [ ] Run generate_raft_data.py — monitor for 20K valid triplets

## Phase 4 — Fine-tuning
- [ ] PROVISION H100 on RunPod or Lambda Labs
- [ ] Run Plan 4.1: training config (checkpoint — confirm hyperparams)
- [ ] Run run_finetuning.py on H100 — Watch WandB for loss curve

## Phase 5 — Evaluation
- [ ] Run Plan 5.1: custom metrics
- [ ] Run Plan 5.2: RAGAS + evaluate.py
- [ ] Run evaluate.py — check all targets pass

## Phase 6 — Inference API
- [ ] Run Plan 6.1: retriever + engine + guardrails
- [ ] Run Plan 6.2: FastAPI + Docker Compose
