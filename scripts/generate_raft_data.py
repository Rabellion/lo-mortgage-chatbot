"""
Generate RAFT triplets from the Pinecone vector store.

Steps:
  1. Pull all chunks from Pinecone
  2. Run async GPT-4o synthesis with progress bar
  3. Validate triplets
  4. Convert to ShareGPT format
  5. Print stats

Usage:
    python scripts/generate_raft_data.py
    python scripts/generate_raft_data.py --target 24000 --questions-per-chunk 6
    python scripts/generate_raft_data.py --skip-synthesis   # validate + convert only
"""

import argparse
import json
import math
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.data.embedding import query_vector_store
from src.raft.dataset import convert_dataset
from src.raft.synthesis import generate_batch
from src.raft.validation import validate_dataset

console = Console()

RAW_TRIPLETS = Path("data/raft_triplets/raw_triplets.jsonl")
VALID_TRIPLETS = Path("data/raft_triplets/valid_triplets.jsonl")
INVALID_TRIPLETS = Path("data/raft_triplets/invalid_triplets.jsonl")
SHAREGPT_OUTPUT = Path("data/raft_triplets/train_sharegpt.json")


def _pull_chunks_from_pinecone(namespace: str, n_chunks: int) -> list[dict]:
    """
    Pull chunks from Pinecone by querying broad mortgage terms.
    Returns list of {text, metadata} dicts.
    """
    console.print(f"[bold]Pulling ~{n_chunks} chunks from Pinecone...[/bold]")

    queries = [
        "mortgage", "home loan", "refinance", "down payment", "credit score",
        "interest rate", "FHA", "VA loan", "Fannie Mae", "closing costs",
        "amortization", "escrow", "conventional loan", "ARM", "fixed rate",
    ]

    seen_ids: set[str] = set()
    chunks: list[dict] = []

    per_query = math.ceil(n_chunks / len(queries)) + 5

    for q in queries:
        if len(chunks) >= n_chunks:
            break
        try:
            results = query_vector_store(q, namespace=namespace, top_k=min(per_query, 100))
            for match in results["matches"]:
                if match["id"] not in seen_ids:
                    seen_ids.add(match["id"])
                    chunks.append({
                        "text": match["text"],
                        "metadata": match["metadata"],
                    })
        except Exception as e:
            console.print(f"[yellow]Query failed for '{q}': {e}[/yellow]")

    console.print(f"[green]Pulled {len(chunks)} unique chunks[/green]")
    return chunks[:n_chunks]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAFT training data")
    parser.add_argument("--target", type=int, default=20000, help="Target valid triplets (default 20000)")
    parser.add_argument("--questions-per-chunk", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--namespace", default="mortgage_docs")
    parser.add_argument("--skip-synthesis", action="store_true", help="Skip synthesis, validate + convert only")
    args = parser.parse_args()

    RAW_TRIPLETS.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_synthesis:
        if not settings.openai_api_key:
            console.print("[bold red]ERROR:[/bold red] OPENAI_API_KEY not set in .env")
            sys.exit(1)

        # Calculate how many chunks we need (with 20% buffer for validation loss)
        target_raw = math.ceil(args.target * 1.2)
        chunks_needed = math.ceil(target_raw / args.questions_per_chunk)
        console.print(
            f"\n[bold]RAFT Synthesis Plan[/bold]\n"
            f"  Target valid triplets : {args.target:,}\n"
            f"  Raw target (×1.2)     : {target_raw:,}\n"
            f"  Questions per chunk   : {args.questions_per_chunk}\n"
            f"  Chunks needed         : {chunks_needed:,}\n"
            f"  Concurrency           : {args.concurrency}\n"
        )

        chunks = _pull_chunks_from_pinecone(args.namespace, chunks_needed)
        if not chunks:
            console.print("[bold red]ERROR:[/bold red] No chunks found in Pinecone. Run build_baseline.py first.")
            sys.exit(1)

        # Progress tracking
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        )

        with progress:
            task = progress.add_task(
                "Generating triplets...",
                total=len(chunks) * args.questions_per_chunk,
            )

            def on_progress(completed: int, total: int) -> None:
                progress.update(task, completed=completed)

            written = generate_batch(
                chunks=chunks,
                output_path=RAW_TRIPLETS,
                questions_per_chunk=args.questions_per_chunk,
                concurrency=args.concurrency,
                progress_callback=on_progress,
            )

        console.print(f"\n[green]Synthesis complete:[/green] {written:,} raw triplets written → {RAW_TRIPLETS}")

    # Validation
    console.rule("[bold blue]Validation")
    stats = validate_dataset(
        input_path=RAW_TRIPLETS,
        valid_output_path=VALID_TRIPLETS,
        invalid_output_path=INVALID_TRIPLETS,
    )

    table = Table("Metric", "Count", show_lines=False)
    table.add_row("Total", str(stats["total"]))
    table.add_row("[green]Valid[/green]", f"[green]{stats['valid']:,}[/green]")
    table.add_row("[red]Invalid[/red]", f"[red]{stats['invalid']:,}[/red]")
    console.print(table)

    if stats["issues"]:
        console.print("\n[bold]Issue breakdown:[/bold]")
        for issue, count in sorted(stats["issues"].items(), key=lambda x: -x[1]):
            console.print(f"  {issue}: {count}")

    if stats["valid"] < args.target:
        console.print(
            f"\n[bold yellow]WARNING:[/bold yellow] Only {stats['valid']:,} valid triplets "
            f"(target: {args.target:,}). "
            "Consider adding more source documents or lowering --target."
        )

    # ShareGPT conversion
    console.rule("[bold blue]ShareGPT Conversion")
    n_converted = convert_dataset(
        input_path=VALID_TRIPLETS,
        output_path=SHAREGPT_OUTPUT,
        fmt="sharegpt",
    )
    console.print(f"[green]Converted {n_converted:,} triplets → {SHAREGPT_OUTPUT}[/green]")

    # Also compute file sizes
    if SHAREGPT_OUTPUT.exists():
        size_mb = SHAREGPT_OUTPUT.stat().st_size / (1024 * 1024)
        console.print(f"Training file size: [bold]{size_mb:.1f} MB[/bold]")

    console.rule("[bold green]Done")
    console.print(
        f"[bold]{stats['valid']:,}[/bold] valid triplets ready for fine-tuning.\n"
        f"Training data: [cyan]{SHAREGPT_OUTPUT}[/cyan]"
    )

    # Verify against FinanceBench triplets file
    with open(VALID_TRIPLETS, encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            meta = rec.get("oracle_metadata", {})
            if meta.get("eval_only"):
                console.print(
                    f"[bold red]ERROR:[/bold red] Line {i+1} has eval_only=True in training set! "
                    "FinanceBench data must never enter training."
                )
                sys.exit(1)

    console.print("[green]FinanceBench contamination check passed.[/green]")


if __name__ == "__main__":
    main()
