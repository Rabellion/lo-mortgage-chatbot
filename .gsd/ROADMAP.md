# ROADMAP.md

> **Current Phase**: 1
> **Milestone**: v1.0 — MVP ML Pipeline

## Must-Haves (from SPEC)

- [ ] Authoritative mortgage document corpus ingested into ChromaDB
- [ ] 20K validated RAFT triplets in ShareGPT format
- [ ] Gemma 4 31B fine-tuned with LoRA on H100
- [ ] Hallucination < 3%, RAGAS faithfulness > 0.85
- [ ] Inference API serving fine-tuned model + RAG

---

## Phases

### Phase 1: Data Acquisition
**Status**: ⬜ Not Started
**Objective**: Populate `data/raw/` with authoritative mortgage documents from all planned sources. This is the hard blocker — nothing downstream works without corpus.
**Deliverables**:
- CFPB mortgage complaint JSON downloaded via API
- FinanceBench JSONL downloaded from GitHub
- Fannie Mae / Freddie Mac / HUD PDFs ingested
- Custom CFPB JSON loader written (not supported by LangChain out of box)
- `pyproject.toml` + project scaffold re-established

### Phase 2: Baseline RAG Pipeline
**Status**: ⬜ Not Started
**Objective**: Build ChromaDB vector store from corpus and verify retrieval quality before RAFT synthesis begins.
**Deliverables**:
- Document ingestion pipeline (PDF, docx, HTML, JSON)
- Text chunking (512 tokens, 50 overlap)
- ChromaDB persistent vector store built and queryable
- Sample retrieval baseline metrics recorded

### Phase 3: RAFT Data Synthesis
**Status**: ⬜ Not Started
**Objective**: Generate 20K+ validated RAFT triplets via concurrent GPT-4o calls. Each triplet = question + oracle doc + distractors + chain-of-thought answer.
**Deliverables**:
- Async synthesis engine (30 concurrent workers, semaphore-controlled)
- Triplet validation pipeline (CoT citation check, rate hallucination check)
- ≥ 20K valid triplets in `data/raft_triplets/valid_triplets.jsonl`
- ShareGPT-format training file `data/raft_triplets/train_sharegpt.json`

### Phase 4: Training Setup & Fine-tuning
**Status**: ⬜ Not Started
**Objective**: Fine-tune Gemma 4 31B on RAFT dataset using Unsloth + TRL on H100. Produce merged 16-bit model.
**Deliverables**:
- Training config locked (LoRA r=16, lr=2e-4, 3 epochs)
- Unsloth + TRL SFTTrainer pipeline wired up
- WandB experiment tracking active
- Fine-tuned merged model saved to disk
- (Optional) Model pushed to HuggingFace Hub

### Phase 5: Evaluation
**Status**: ⬜ Not Started
**Objective**: Measure fine-tuned model against success criteria. If metrics miss, identify gaps for Run #2.
**Deliverables**:
- RAGAS evaluation (faithfulness, answer relevancy, context precision/recall)
- Custom hallucination rate metric (fabricated rates/figures)
- Custom citation coverage metric
- Tone score evaluation
- Evaluation report written to `data/evaluation/results.json`

### Phase 6: Inference API
**Status**: ⬜ Not Started
**Objective**: Wire fine-tuned model into FastAPI + vLLM serving stack with RAG at inference. Owned by API team but ML provides the model + retriever interface.
**Deliverables**:
- vLLM model server config for fine-tuned Gemma 4 31B
- RAG retriever integrated at inference (top-5 ChromaDB results)
- Guardrails: off-topic blocking, fabricated rate detection, disclaimer injection
- FastAPI endpoints: `/api/chat/` (REST) + `/api/chat/stream` (SSE)
- Lead capture endpoint: `/api/leads/`
- Docker Compose: API + Redis + ChromaDB

---

*Last updated: 2026-04-04*
