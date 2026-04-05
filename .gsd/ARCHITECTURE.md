# ARCHITECTURE.md — System Design

## ML Pipeline (Huzaifa's scope)

```
data/raw/  (PDFs, JSON, JSONL)
    │
    ▼
[Phase 1] Data Acquisition
    │  download_datasets.py — CFPB API + FinanceBench
    │  Manual PDFs: Fannie Mae, Freddie Mac, HUD 4000.1
    │
    ▼
[Phase 2] Baseline RAG
    │  src/data/ingestion.py    — LangChain loaders (PDF, docx, HTML, JSON)
    │  src/data/chunking.py     — RecursiveCharacterTextSplitter, 512t/50 overlap
    │  src/data/embedding.py    — sentence-transformers all-MiniLM-L6-v2 → ChromaDB
    │  chroma_db/               — persistent vector store
    │
    ▼
[Phase 3] RAFT Synthesis
    │  src/raft/synthesis.py    — AsyncOpenAI, 30 concurrent workers, semaphore
    │  src/raft/validation.py   — CoT citation check, rate hallucination check
    │  src/raft/dataset.py      — ShareGPT / Alpaca conversion
    │  data/raft_triplets/      — raw_triplets.jsonl → valid_triplets.jsonl → train_sharegpt.json
    │
    ▼
[Phase 4] Fine-tuning (H100 80GB)
    │  src/training/config.py   — LoRA r=16, lr=2e-4, 3 epochs, bf16
    │  src/training/finetune.py — Unsloth FastLanguageModel + TRL SFTTrainer
    │  models/gemma-4-31b-raft/ — merged 16-bit output
    │
    ▼
[Phase 5] Evaluation
    │  src/evaluation/metrics.py    — hallucination rate, citation coverage
    │  src/evaluation/ragas_eval.py — faithfulness, relevancy, precision, recall
    │  data/evaluation/results.json — final metrics
    │
    ▼
[Phase 6] Inference API (API team, ML provides model + retriever)
    │  src/inference/engine.py    — RAG: retrieve top-5 → format context → vLLM
    │  src/inference/retriever.py — ChromaDB vector search
    │  src/inference/guardrails.py — off-topic block, rate hallucination detect, disclaimer
    │  src/api/main.py            — FastAPI app
    │  src/api/routes/chat.py     — REST + SSE streaming
    │  src/api/routes/leads.py    — lead capture
```

## Key Technology Choices

| Decision | Choice | Reason |
|----------|--------|--------|
| Base model | Gemma 4 31B Dense | Apache 2.0, strong instruction following, fits H100 80GB |
| Fine-tuning | LoRA 16-bit via Unsloth | Memory-efficient, no QLoRA degradation risk on 31B dense |
| Synthesis LLM | GPT-4o | Highest quality CoT generation for training data |
| Vector DB | ChromaDB | Lightweight persistent, no infra overhead |
| Embeddings | all-MiniLM-L6-v2 | Fast, good retrieval quality for this domain |
| Async synthesis | asyncio + semaphore | 30× speed-up vs sequential |

## Data Flow at Inference

```
User query
    → Guardrails (off-topic check)
    → ChromaDB retriever (top-5 chunks)
    → Context formatter (oracle + distractors)
    → vLLM (fine-tuned Gemma 4 31B)
    → Response guardrails (rate check, disclaimer)
    → Client (REST or SSE)
```
