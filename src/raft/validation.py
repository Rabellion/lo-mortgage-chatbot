"""
RAFT triplet validation.

Validates triplets for:
- Structural completeness
- CoT citation markers
- Rate/figure fabrication (any % in answer must appear verbatim in oracle_doc)
"""

import json
import re
from pathlib import Path
from typing import Any

RATE_PATTERN = re.compile(r"\d+\.?\d*\s*%")


def validate_triplet(triplet: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate a single RAFT triplet.

    Returns (True, []) if valid, (False, [issue, ...]) otherwise.
    """
    issues: list[str] = []

    question = triplet.get("question", "")
    oracle_doc = triplet.get("oracle_doc", "")
    answer = triplet.get("answer", "")
    cot = triplet.get("chain_of_thought", "")
    distractors = triplet.get("distractor_docs", [])

    # Structural checks
    if not question or len(question) < 10:
        issues.append("question_too_short")
    if not oracle_doc or len(oracle_doc) < 20:
        issues.append("oracle_doc_too_short")
    if not answer or len(answer) < 10:
        issues.append("answer_too_short")

    # CoT citation check
    if "##begin_quote##" not in cot or "##end_quote##" not in cot:
        issues.append("missing_cot_citations")

    # Distractor check
    if not distractors or len(distractors) < 1:
        issues.append("no_distractors")

    # Rate fabrication check
    rates_in_answer = RATE_PATTERN.findall(answer)
    for rate in rates_in_answer:
        # Normalise whitespace for comparison
        normalized = rate.replace(" ", "")
        oracle_normalized = oracle_doc.replace(" ", "")
        if normalized not in oracle_normalized:
            issues.append(f"fabricated_rate:{rate.strip()}")

    return (len(issues) == 0, issues)


def validate_dataset(
    input_path: str | Path,
    valid_output_path: str | Path | None = None,
    invalid_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Stream through a JSONL file and validate every triplet.

    Returns stats dict:
        {total, valid, invalid, issues: {issue_str: count}}
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    valid_f = None
    invalid_f = None

    if valid_output_path:
        valid_output_path = Path(valid_output_path)
        valid_output_path.parent.mkdir(parents=True, exist_ok=True)
        valid_f = open(valid_output_path, "w", encoding="utf-8")

    if invalid_output_path:
        invalid_output_path = Path(invalid_output_path)
        invalid_output_path.parent.mkdir(parents=True, exist_ok=True)
        invalid_f = open(invalid_output_path, "w", encoding="utf-8")

    total = 0
    valid_count = 0
    invalid_count = 0
    issue_counts: dict[str, int] = {}

    try:
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    triplet = json.loads(line)
                except json.JSONDecodeError:
                    issue_counts["json_parse_error"] = issue_counts.get("json_parse_error", 0) + 1
                    invalid_count += 1
                    total += 1
                    continue

                total += 1
                ok, issues = validate_triplet(triplet)

                if ok:
                    valid_count += 1
                    if valid_f:
                        valid_f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                else:
                    invalid_count += 1
                    for issue in issues:
                        issue_counts[issue] = issue_counts.get(issue, 0) + 1
                    if invalid_f:
                        invalid_f.write(
                            json.dumps({"triplet": triplet, "issues": issues}, ensure_ascii=False) + "\n"
                        )
    finally:
        if valid_f:
            valid_f.close()
        if invalid_f:
            invalid_f.close()

    return {
        "total": total,
        "valid": valid_count,
        "invalid": invalid_count,
        "issues": issue_counts,
    }
