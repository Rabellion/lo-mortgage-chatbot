"""
Convert validated RAFT triplets to training formats.

Supported formats:
  - sharegpt: for Unsloth SFTTrainer
  - alpaca: standard instruction-following format
"""

import json
import random
from pathlib import Path
from typing import Any

SYSTEM_MESSAGE = (
    "You are a knowledgeable and friendly mortgage advisor. "
    "Answer the client's question using ONLY the provided documents. "
    "Cite the relevant document when answering. "
    "If the documents don't contain the answer, say so honestly. "
    "Never fabricate rates, figures, or eligibility criteria. "
    "Speak in warm, jargon-light conversational English."
)


def _format_docs(oracle_doc: str, distractor_docs: list[str]) -> str:
    """Combine oracle + distractors into shuffled <DOCUMENT> blocks."""
    all_docs = [oracle_doc] + distractor_docs
    random.shuffle(all_docs)
    blocks = []
    for i, doc in enumerate(all_docs, start=1):
        blocks.append(f"<DOCUMENT id={i}>\n{doc}\n</DOCUMENT>")
    return "\n\n".join(blocks)


def triplet_to_sharegpt(triplet: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a RAFT triplet to ShareGPT format for Unsloth SFTTrainer.

    Returns:
        {
          "conversations": [
            {"from": "system", "value": ...},
            {"from": "human", "value": "<docs>\n\n<question>"},
            {"from": "gpt", "value": "<chain_of_thought>\n\n<answer>"}
          ]
        }
    """
    doc_context = _format_docs(
        triplet["oracle_doc"],
        triplet.get("distractor_docs", []),
    )
    human_turn = f"{doc_context}\n\nQuestion: {triplet['question']}"
    gpt_turn = f"{triplet['chain_of_thought']}\n\n{triplet['answer']}"

    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_MESSAGE},
            {"from": "human", "value": human_turn},
            {"from": "gpt", "value": gpt_turn},
        ]
    }


def triplet_to_alpaca(triplet: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a RAFT triplet to Alpaca instruction format.

    Returns: {instruction, input, output}
    """
    doc_context = _format_docs(
        triplet["oracle_doc"],
        triplet.get("distractor_docs", []),
    )
    return {
        "instruction": triplet["question"],
        "input": doc_context,
        "output": f"{triplet['chain_of_thought']}\n\n{triplet['answer']}",
    }


def convert_dataset(
    input_path: str | Path,
    output_path: str | Path,
    fmt: str = "sharegpt",
) -> int:
    """
    Stream JSONL triplets and convert to training format JSON array.

    Args:
        input_path: path to validated_triplets.jsonl
        output_path: path to write (train_sharegpt.json or train_alpaca.json)
        fmt: "sharegpt" or "alpaca"

    Returns:
        Number of examples written.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt not in ("sharegpt", "alpaca"):
        raise ValueError(f"Unknown format: {fmt}. Must be 'sharegpt' or 'alpaca'.")

    converter = triplet_to_sharegpt if fmt == "sharegpt" else triplet_to_alpaca

    examples = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                triplet = json.loads(line)
                examples.append(converter(triplet))
            except (json.JSONDecodeError, KeyError):
                continue

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    return len(examples)
