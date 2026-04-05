"""
Text chunking pipeline.

Splits documents into uniform chunks using RecursiveCharacterTextSplitter.
eval_only documents are filtered out before chunking — they never enter training.
"""

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

from src.config import settings

console = Console()


def chunk_documents(
    docs: list[dict[str, Any]],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict[str, Any]]:
    """
    Chunk training documents.

    Filters out eval_only docs, then splits remaining docs into chunks.
    Returns list of dicts: {"text": str, "metadata": dict}.
    Each chunk inherits parent metadata + "chunk_index" field.
    Chunks shorter than 50 characters are discarded as noise.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    # Filter eval_only
    training_docs = [d for d in docs if not d.get("metadata", {}).get("eval_only")]
    skipped = len(docs) - len(training_docs)
    if skipped:
        console.print(f"[yellow]Skipping {skipped} eval-only docs (FinanceBench)[/yellow]")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, Any]] = []
    for doc in training_docs:
        text = doc.get("text", "")
        if not text.strip():
            continue

        parts = splitter.split_text(text)
        for idx, part in enumerate(parts):
            if len(part.strip()) < 50:
                continue
            chunks.append({
                "text": part,
                "metadata": {
                    **doc.get("metadata", {}),
                    "chunk_index": idx,
                },
            })

    console.print(
        f"[green]Chunked {len(training_docs)} docs → {len(chunks)} chunks[/green] "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks
