---
phase: 2
plan: 1
wave: 1
---

# Plan 2.1: Document Ingestion + Chunking Pipeline

## Objective
Build the ingestion and chunking layer that normalizes all source formats (PDF, docx, HTML, CFPB JSON) into uniformly chunked text ready for embedding.

## Context
- .gsd/SPEC.md
- .gsd/ARCHITECTURE.md
- src/config.py
- src/data/loaders/cfpb.py

## Tasks

<task type="auto">
  <name>Write unified document ingestion module</name>
  <files>
    src/data/ingestion.py
  </files>
  <action>
    Write src/data/ingestion.py with:

    LOADER_MAP dict mapping extensions to LangChain loaders:
      .pdf → PyPDFLoader
      .docx / .doc → UnstructuredWordDocumentLoader
      .html / .htm → UnstructuredHTMLLoader
      (CFPB JSON handled separately — not via DirectoryLoader)

    load_documents(data_dir: str | Path) -> list:
      - Use DirectoryLoader with multithreading for PDF/docx/HTML
      - Detect if cfpb_mortgage_complaints.json exists and load via CFPBComplaintLoader
      - Detect if financebench.jsonl exists: parse each line as JSON, use
        question + answer fields as page_content, metadata source_type="financebench"
        (FinanceBench is eval-only: tag metadata["eval_only"] = True)
      - Tag all docs with source_type in metadata
      - Return combined list

    load_single_file(file_path: str | Path) -> list:
      - Dispatch to correct loader by extension
      - Support .json via CFPBComplaintLoader

    DO NOT load FinanceBench docs into the training vector store —
    eval_only=True docs must be filtered out before embedding.
  </action>
  <verify>python -c "from src.data.ingestion import load_documents; docs = load_documents('data/raw'); train = [d for d in docs if not d.metadata.get('eval_only')]; print(f'{len(train)} training docs, {len(docs)-len(train)} eval-only')"</verify>
  <done>Prints counts without error; training doc count is positive</done>
</task>

<task type="auto">
  <name>Write chunking module</name>
  <files>
    src/data/chunking.py
  </files>
  <action>
    Write src/data/chunking.py with:

    chunk_documents(docs: list, chunk_size: int = None, chunk_overlap: int = None) -> list[dict]:
      - Default chunk_size and chunk_overlap from settings (512, 50)
      - Use LangChain RecursiveCharacterTextSplitter
      - Returns list of dicts: {"text": str, "metadata": dict}
      - Preserve all metadata from parent document
      - Add "chunk_index" to metadata for traceability
      - Skip chunks shorter than 50 characters (noise)

    DO NOT chunk eval_only documents — filter them out before chunking.
    Chunk size is in characters not tokens (LangChain default); keep at 2000 chars
    to approximate 512 tokens for this domain.
  </action>
  <verify>python -c "from src.data.ingestion import load_documents; from src.data.chunking import chunk_documents; docs = load_documents('data/raw'); chunks = chunk_documents([d for d in docs if not d.metadata.get('eval_only')]); print(len(chunks), 'chunks'); assert all('text' in c and 'metadata' in c for c in chunks)"</verify>
  <done>Prints positive chunk count; each chunk is a dict with 'text' and 'metadata' keys</done>
</task>
