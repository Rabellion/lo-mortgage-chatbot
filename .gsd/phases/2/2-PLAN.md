---
phase: 2
plan: 2
wave: 2
---

# Plan 2.2: ChromaDB Vector Store + Baseline Script

## Objective
Embed all training chunks into a persistent ChromaDB collection and verify retrieval quality with sample mortgage questions. This is the baseline RAG that RAFT synthesis and inference both depend on.

## Context
- .gsd/SPEC.md
- .gsd/ARCHITECTURE.md
- src/config.py
- src/data/ingestion.py
- src/data/chunking.py

## Tasks

<task type="auto">
  <name>Write ChromaDB embedding module</name>
  <files>
    src/data/embedding.py
  </files>
  <action>
    Write src/data/embedding.py with:

    get_chroma_client() -> chromadb.Client:
      - PersistentClient at settings.chroma_persist_dir

    build_vector_store(chunks: list[dict], collection_name: str = "mortgage_docs") -> Collection:
      - Use sentence-transformers/all-MiniLM-L6-v2 via chromadb's SentenceTransformerEmbeddingFunction
      - Get or create collection with cosine distance
      - Batch upsert in groups of 500 (ChromaDB limit)
      - Generate deterministic IDs: sha256(text)[:16]
      - Store chunk text as document, metadata as metadata
      - Print progress with Rich

    query_vector_store(query: str, collection_name: str = "mortgage_docs", top_k: int = 5) -> dict:
      - Query collection, return ChromaDB results dict (documents, metadatas, distances)

    DO NOT re-embed chunks that already exist (upsert handles dedup by ID).
    DO NOT store eval_only chunks in the mortgage_docs collection.
  </action>
  <verify>python -c "from src.data.embedding import get_chroma_client; c = get_chroma_client(); print('ChromaDB OK')"</verify>
  <done>Prints ChromaDB OK without error</done>
</task>

<task type="auto">
  <name>Write build_baseline.py script</name>
  <files>
    scripts/build_baseline.py
  </files>
  <action>
    Write scripts/build_baseline.py that:
    1. Loads documents from data/raw/ (excluding eval_only)
    2. Chunks them
    3. Builds ChromaDB vector store
    4. Runs 5 sample mortgage questions and prints retrieval results in a Rich table:
       - "What credit score do I need for a conventional mortgage?"
       - "How much down payment is required for a first-time home buyer?"
       - "What is the difference between a fixed-rate and adjustable-rate mortgage?"
       - "Can I refinance if my home is underwater?"
       - "What documents do I need to apply for a mortgage?"
    5. Prints cosine similarity score for top-1 result per question
    6. Aborts with clear error if data/raw/ is empty

    This script is idempotent — safe to re-run (upsert deduplication).
  </action>
  <verify>python scripts/build_baseline.py 2>&1 | tail -5</verify>
  <done>Script prints "Baseline RAG setup complete" and a table with 5 questions and similarity scores</done>
</task>
