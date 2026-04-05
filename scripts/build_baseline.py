"""
Build baseline RAG pipeline and verify retrieval quality.

Steps:
  1. Load documents from data/raw/
  2. Chunk training docs
  3. Embed and upsert into Pinecone
  4. Run 5 sample queries and print retrieval results

Usage:
    python scripts/build_baseline.py
    python scripts/build_baseline.py --skip-build   # query only (index already built)
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.data.chunking import chunk_documents
from src.data.embedding import build_vector_store, query_vector_store
from src.data.ingestion import load_documents

console = Console()

SAMPLE_QUESTIONS = [
    "What credit score do I need for a conventional mortgage?",
    "How much down payment is required for a first-time home buyer?",
    "What is the difference between a fixed-rate and adjustable-rate mortgage?",
    "Can I refinance if my home is underwater?",
    "What documents do I need to apply for a mortgage?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build baseline RAG and verify retrieval")
    parser.add_argument("--skip-build", action="store_true", help="Skip ingestion/embedding, query only")
    parser.add_argument("--namespace", default="mortgage_docs", help="Pinecone namespace")
    args = parser.parse_args()

    settings.ensure_dirs()

    if not args.skip_build:
        # Check data exists
        raw_dir = settings.data_raw_dir
        data_files = list(raw_dir.glob("*"))
        if not data_files:
            console.print(
                "[bold red]ERROR:[/bold red] data/raw/ is empty. "
                "Run [bold]python scripts/download_datasets.py[/bold] first."
            )
            sys.exit(1)

        # Load
        docs = load_documents(raw_dir)
        training_docs = [d for d in docs if not d["metadata"].get("eval_only")]
        if not training_docs:
            console.print("[bold red]ERROR:[/bold red] No training documents found (all eval_only or empty).")
            sys.exit(1)

        # Chunk
        chunks = chunk_documents(training_docs)
        console.print(f"[bold]{len(chunks)}[/bold] chunks ready for embedding\n")

        # Embed + upsert
        build_vector_store(chunks, namespace=args.namespace)

    # Sample retrieval
    console.rule("[bold blue]Retrieval Quality Check")
    table = Table(
        "Question",
        "Top-1 Score",
        "Top-1 Source",
        "Snippet",
        show_lines=True,
    )

    for q in SAMPLE_QUESTIONS:
        results = query_vector_store(q, namespace=args.namespace, top_k=1)
        if not results["matches"]:
            table.add_row(q, "—", "—", "[dim]no results[/dim]")
            continue
        top = results["matches"][0]
        score = f"{top['score']:.4f}"
        source = top["metadata"].get("source_type", "unknown")
        snippet = top["text"][:120].replace("\n", " ") + "..."
        table.add_row(q, score, source, snippet)

    console.print(table)
    console.print("\n[bold green]Baseline RAG setup complete.[/bold green]")
    console.print(f"Pinecone index: [cyan]{settings.pinecone_index_name}[/cyan]")


if __name__ == "__main__":
    main()
