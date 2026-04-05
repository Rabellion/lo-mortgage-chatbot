---
phase: 1
plan: 1
wave: 1
---

# Plan 1.1: Project Scaffold + Auto-Download Datasets

## Objective
Re-establish the Python project scaffold (pyproject.toml, directory structure) and auto-download the two publicly available datasets (CFPB + FinanceBench) into `data/raw/`. This unblocks all downstream phases.

## Context
- .gsd/SPEC.md
- .gsd/ARCHITECTURE.md

## Tasks

<task type="auto">
  <name>Re-establish project scaffold</name>
  <files>
    pyproject.toml
    .env.example
    data/raw/.gitkeep
    data/processed/.gitkeep
    data/raft_triplets/.gitkeep
    data/evaluation/.gitkeep
    src/__init__.py
    src/config.py
  </files>
  <action>
    Create pyproject.toml with Python 3.10+, all dependencies:
      Core: fastapi, uvicorn, openai, google-generativeai, chromadb, sentence-transformers,
            langchain, langchain-community, pypdf, python-docx, beautifulsoup4, httpx,
            pydantic-settings, python-dotenv, rich, redis
      Training (optional group): torch>=2.4, unsloth, trl, transformers, accelerate,
            bitsandbytes>=0.44, datasets>=3.0, wandb, peft
      Eval (optional group): ragas, deepeval
      Dev: pytest, pytest-asyncio, ruff

    Create src/config.py with Pydantic BaseSettings reading from .env:
      openai_api_key, google_api_key, chroma_persist_dir, vllm_base_url, model_name,
      redis_url, wandb_api_key, wandb_project, hf_token, app_env, app_port, log_level,
      chunk_size=512, chunk_overlap=50, embedding_model, retriever_top_k=5

    Create .env.example with all keys documented.
    Create data/ subdirectory .gitkeep files.

    DO NOT write any implementation logic yet — config and scaffold only.
  </action>
  <verify>python -c "from src.config import settings; print(settings.chunk_size)"</verify>
  <done>Command prints 512 without error</done>
</task>

<task type="auto">
  <name>Write and run dataset download script</name>
  <files>
    scripts/download_datasets.py
  </files>
  <action>
    Write scripts/download_datasets.py that:
    1. Downloads CFPB mortgage complaints via their public API:
       URL: https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/
       Params: product=Mortgage, size=1000, sort=created_date_desc, no_aggs=True, format=json
       Save to: data/raw/cfpb_mortgage_complaints.json

    2. Downloads FinanceBench from GitHub:
       URL: https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl
       Save to: data/raw/financebench.jsonl

    Use httpx with 60s timeout. Print counts of records downloaded using Rich console.
    Print reminder at end: "Place Fannie Mae, Freddie Mac, HUD 4000.1 PDFs in data/raw/ manually"
  </action>
  <verify>python scripts/download_datasets.py && ls data/raw/</verify>
  <done>Both cfpb_mortgage_complaints.json and financebench.jsonl exist in data/raw/ with non-zero size</done>
</task>
