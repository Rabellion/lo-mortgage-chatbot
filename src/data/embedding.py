"""
Pinecone vector store for mortgage document chunks.

Handles:
- Creating / connecting to a Pinecone index
- Batch upserting chunks with sentence-transformer embeddings
- Querying top-K nearest neighbours
"""

import hashlib
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sentence_transformers import SentenceTransformer

from src.config import settings

console = Console()

_UPSERT_BATCH = 200  # Pinecone recommended batch size


def _get_index():
    """Return a connected Pinecone Index, creating it if needed."""
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=settings.pinecone_api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        console.print(
            f"[bold]Creating Pinecone index[/bold] [cyan]{settings.pinecone_index_name}[/cyan] ..."
        )
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=384,  # all-MiniLM-L6-v2 output dim
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        console.print("[green]Index created.[/green]")
    else:
        console.print(
            f"[green]Connected to existing index:[/green] {settings.pinecone_index_name}"
        )

    return pc.Index(settings.pinecone_index_name)


def _chunk_id(text: str) -> str:
    """Deterministic 16-char ID from chunk content."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def build_vector_store(
    chunks: list[dict[str, Any]],
    namespace: str = "mortgage_docs",
) -> Any:
    """
    Embed chunks and upsert into Pinecone.

    Uses all-MiniLM-L6-v2 embeddings (384-dim, cosine).
    Upserts in batches of 200. Returns the Pinecone Index object.
    eval_only chunks must be filtered BEFORE calling this function.
    """
    if not chunks:
        console.print("[yellow]No chunks to embed.[/yellow]")
        return None

    console.print(f"\n[bold]Embedding {len(chunks)} chunks[/bold] with {settings.embedding_model}...")
    model = SentenceTransformer(settings.embedding_model)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    index = _get_index()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Upserting to Pinecone...", total=len(chunks))

        for start in range(0, len(chunks), _UPSERT_BATCH):
            batch_chunks = chunks[start : start + _UPSERT_BATCH]
            batch_embs = embeddings[start : start + _UPSERT_BATCH]

            vectors = []
            for chunk, emb in zip(batch_chunks, batch_embs):
                # Pinecone metadata values must be str/int/float/bool/list
                meta = {}
                for k, v in chunk.get("metadata", {}).items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    else:
                        meta[k] = str(v)
                meta["text"] = chunk["text"][:1000]  # store first 1K chars for retrieval

                vectors.append({
                    "id": _chunk_id(chunk["text"]),
                    "values": emb.tolist(),
                    "metadata": meta,
                })

            index.upsert(vectors=vectors, namespace=namespace)
            progress.update(task, advance=len(batch_chunks))

    console.print(f"[green]Upserted {len(chunks)} vectors → namespace '{namespace}'[/green]")
    return index


def query_vector_store(
    query: str,
    namespace: str = "mortgage_docs",
    top_k: int | None = None,
) -> dict[str, Any]:
    """
    Query the Pinecone index for the most relevant chunks.

    Returns dict with keys: matches (list of {id, score, metadata}).
    """
    top_k = top_k or settings.top_k
    model = SentenceTransformer(settings.embedding_model)
    query_emb = model.encode([query], convert_to_numpy=True)[0].tolist()

    index = _get_index()
    results = index.query(
        vector=query_emb,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )
    return {
        "matches": [
            {
                "id": m.id,
                "score": m.score,
                "metadata": m.metadata,
                "text": m.metadata.get("text", ""),
            }
            for m in results.matches
        ]
    }
