"""
Download all raw datasets needed for the LO Mortgage Chatbot pipeline.

Sources:
  1. CFPB Consumer Complaints — public API, mortgage category
  2. FinanceBench — GitHub JSONL release (eval only)

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --cfpb-limit 5000
    python scripts/download_datasets.py --skip-cfpb
    python scripts/download_datasets.py --skip-financebench
"""

import argparse
import json
import sys
from pathlib import Path

import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import settings

console = Console()

CFPB_API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
CFPB_OUTPUT = settings.data_raw_dir / "cfpb_complaints.json"

FINANCEBENCH_URL = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl"
)
FINANCEBENCH_OUTPUT = settings.data_raw_dir / "financebench.jsonl"


def download_cfpb(limit: int = 10000) -> None:
    """Download mortgage complaints from CFPB public API (v1 search endpoint)."""
    console.rule("[bold blue]CFPB Complaints")

    if CFPB_OUTPUT.exists():
        console.print(f"[yellow]Already exists:[/yellow] {CFPB_OUTPUT} — skipping. Delete to re-download.")
        return

    console.print(f"Fetching up to [bold]{limit}[/bold] mortgage complaints from CFPB API...")

    # The v1 search API uses 'frm' (not 'from') for offset and returns a flat list of hits.
    # Page size capped at 100 (API enforced).
    PAGE_SIZE = 100
    all_hits = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading complaints...", total=limit)

        while len(all_hits) < limit:
            params = {
                "product": "Mortgage",
                "has_narrative": "true",
                "size": min(PAGE_SIZE, limit - len(all_hits)),
                "frm": len(all_hits),
                "sort": "created_date_desc",
                "format": "json",
                "no_aggs": "true",
            }

            resp = requests.get(CFPB_API_URL, params=params, timeout=30)
            resp.raise_for_status()

            data = resp.json()

            # v1 API returns a flat list of hit objects
            if isinstance(data, list):
                hits = data
            else:
                # fallback: try nested structure
                hits = data.get("hits", {}).get("hits", [])

            if not hits:
                break

            all_hits.extend(hits)
            progress.update(task, completed=len(all_hits))

            if len(hits) < PAGE_SIZE:
                break  # last page

    all_hits = all_hits[:limit]
    CFPB_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CFPB_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_hits, f, ensure_ascii=False, indent=2)

    console.print(f"[green]Saved {len(all_hits)} complaints →[/green] {CFPB_OUTPUT}")


def download_financebench() -> None:
    """Download FinanceBench JSONL from GitHub (eval only — never enters training)."""
    console.rule("[bold blue]FinanceBench")

    if FINANCEBENCH_OUTPUT.exists():
        console.print(f"[yellow]Already exists:[/yellow] {FINANCEBENCH_OUTPUT} — skipping.")
        return

    console.print("Downloading FinanceBench from GitHub...")

    resp = requests.get(FINANCEBENCH_URL, timeout=60, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    FINANCEBENCH_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading...", total=total or None)
        with open(FINANCEBENCH_OUTPUT, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                progress.update(task, advance=len(chunk))

    # Count lines and verify
    line_count = sum(1 for _ in open(FINANCEBENCH_OUTPUT, encoding="utf-8"))
    console.print(f"[green]Saved {line_count} FinanceBench records →[/green] {FINANCEBENCH_OUTPUT}")
    console.print("[dim]Note: FinanceBench is eval_only=True — it will NOT enter the training vector store.[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw datasets for LO Mortgage Chatbot")
    parser.add_argument("--cfpb-limit", type=int, default=10000, help="Max CFPB complaints to download")
    parser.add_argument("--skip-cfpb", action="store_true", help="Skip CFPB download")
    parser.add_argument("--skip-financebench", action="store_true", help="Skip FinanceBench download")
    args = parser.parse_args()

    settings.ensure_dirs()
    console.print("[bold green]LO Mortgage Chatbot — Dataset Downloader[/bold green]\n")

    if not args.skip_cfpb:
        download_cfpb(limit=args.cfpb_limit)

    if not args.skip_financebench:
        download_financebench()

    console.rule("[bold green]Done")
    console.print(f"Raw data directory: [bold]{settings.data_raw_dir.resolve()}[/bold]")


if __name__ == "__main__":
    main()
