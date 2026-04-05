"""
Document ingestion pipeline.

Loads all source formats (PDF, docx, HTML, CFPB JSON, FinanceBench JSONL)
into a uniform list of dicts: {"text": str, "metadata": dict}.

FinanceBench docs are tagged eval_only=True and must NEVER enter the training store.
"""

import json
from pathlib import Path
from typing import Any

from rich.console import Console

from src.data.loaders.cfpb import CFPBComplaintLoader

console = Console()

CFPB_FILENAME = "cfpb_complaints.json"
FINANCEBENCH_FILENAME = "financebench.jsonl"


def _load_pdfs(data_dir: Path) -> list[dict[str, Any]]:
    from langchain_community.document_loaders import PyPDFLoader

    docs = []
    for pdf_path in data_dir.glob("**/*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            for page in pages:
                if not page.page_content.strip():
                    continue
                docs.append({
                    "text": page.page_content,
                    "metadata": {
                        **page.metadata,
                        "source_type": "pdf",
                        "eval_only": False,
                    },
                })
            console.print(f"  [green]PDF:[/green] {pdf_path.name} → {len(pages)} pages")
        except Exception as e:
            console.print(f"  [red]Failed to load {pdf_path.name}:[/red] {e}")
    return docs


def _load_docx(data_dir: Path) -> list[dict[str, Any]]:
    try:
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    except ImportError:
        console.print("[yellow]UnstructuredWordDocumentLoader not available — skipping .docx files[/yellow]")
        return []

    docs = []
    for docx_path in data_dir.glob("**/*.doc*"):
        try:
            loader = UnstructuredWordDocumentLoader(str(docx_path))
            pages = loader.load()
            for page in pages:
                if not page.page_content.strip():
                    continue
                docs.append({
                    "text": page.page_content,
                    "metadata": {
                        **page.metadata,
                        "source_type": "docx",
                        "eval_only": False,
                    },
                })
            console.print(f"  [green]DOCX:[/green] {docx_path.name} → {len(pages)} sections")
        except Exception as e:
            console.print(f"  [red]Failed to load {docx_path.name}:[/red] {e}")
    return docs


def _load_html(data_dir: Path) -> list[dict[str, Any]]:
    try:
        from langchain_community.document_loaders import UnstructuredHTMLLoader
    except ImportError:
        console.print("[yellow]UnstructuredHTMLLoader not available — skipping .html files[/yellow]")
        return []

    docs = []
    for html_path in data_dir.glob("**/*.htm*"):
        try:
            loader = UnstructuredHTMLLoader(str(html_path))
            pages = loader.load()
            for page in pages:
                if not page.page_content.strip():
                    continue
                docs.append({
                    "text": page.page_content,
                    "metadata": {
                        **page.metadata,
                        "source_type": "html",
                        "eval_only": False,
                    },
                })
        except Exception as e:
            console.print(f"  [red]Failed to load {html_path.name}:[/red] {e}")
    return docs


def _load_cfpb(data_dir: Path) -> list[dict[str, Any]]:
    cfpb_path = data_dir / CFPB_FILENAME
    if not cfpb_path.exists():
        console.print(f"  [yellow]CFPB file not found:[/yellow] {cfpb_path} — skipping")
        return []

    loader = CFPBComplaintLoader(cfpb_path)
    docs = loader.load()
    console.print(f"  [green]CFPB:[/green] {len(docs)} complaint narratives loaded")
    return docs


def _load_financebench(data_dir: Path) -> list[dict[str, Any]]:
    fb_path = data_dir / FINANCEBENCH_FILENAME
    if not fb_path.exists():
        console.print(f"  [yellow]FinanceBench file not found:[/yellow] {fb_path} — skipping")
        return []

    docs = []
    with open(fb_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                question = record.get("question", "")
                answer = record.get("answer", record.get("long_answer", ""))
                if not question or not answer:
                    continue
                docs.append({
                    "text": f"Q: {question}\nA: {answer}",
                    "metadata": {
                        "source_type": "financebench",
                        "eval_only": True,
                        "question_id": record.get("question_id", ""),
                        "doc_name": record.get("doc_name", ""),
                    },
                })
            except json.JSONDecodeError:
                continue

    console.print(
        f"  [green]FinanceBench:[/green] {len(docs)} records loaded [dim](eval_only=True)[/dim]"
    )
    return docs


def load_documents(data_dir: str | Path) -> list[dict[str, Any]]:
    """
    Load all documents from data_dir.

    Returns combined list of dicts with keys: text, metadata.
    FinanceBench docs have metadata["eval_only"] = True.
    All other docs have eval_only = False.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    console.rule("[bold blue]Loading documents")

    all_docs: list[dict[str, Any]] = []
    all_docs.extend(_load_cfpb(data_dir))
    all_docs.extend(_load_pdfs(data_dir))
    all_docs.extend(_load_docx(data_dir))
    all_docs.extend(_load_html(data_dir))
    all_docs.extend(_load_financebench(data_dir))

    train = [d for d in all_docs if not d["metadata"].get("eval_only")]
    eval_only = [d for d in all_docs if d["metadata"].get("eval_only")]
    console.print(
        f"\n[bold]Total:[/bold] {len(all_docs)} docs "
        f"({len(train)} training, {len(eval_only)} eval-only)"
    )
    return all_docs


def load_single_file(file_path: str | Path) -> list[dict[str, Any]]:
    """Load a single file, dispatching by extension."""
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return _load_pdfs(file_path.parent)
    elif ext in (".doc", ".docx"):
        return _load_docx(file_path.parent)
    elif ext in (".html", ".htm"):
        return _load_html(file_path.parent)
    elif ext == ".json":
        loader = CFPBComplaintLoader(file_path)
        return loader.load()
    elif ext == ".jsonl":
        return _load_financebench(file_path.parent)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
