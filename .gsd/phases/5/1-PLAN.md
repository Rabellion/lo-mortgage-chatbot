---
phase: 5
plan: 1
wave: 1
---

# Plan 5.1: Custom Evaluation Metrics

## Objective
Build custom evaluation metrics for hallucination rate and citation coverage. These are domain-specific and more meaningful than generic NLP metrics for a financial chatbot where fabricated rates = legal risk.

## Context
- .gsd/SPEC.md (success criteria: hallucination < 3%, citation coverage > 90%)
- .gsd/ARCHITECTURE.md
- src/config.py

## Tasks

<task type="auto">
  <name>Write custom metrics module</name>
  <files>
    src/evaluation/metrics.py
    src/evaluation/__init__.py
  </files>
  <action>
    Write src/evaluation/metrics.py with:

    detect_hallucinated_rates(answer: str, context_docs: list[str]) -> list[str]:
      - Find all rate patterns in answer: r"\b\d+\.?\d*\s*%"
      - For each rate, check if it appears verbatim in any context_doc
      - Return list of rates NOT found in any context doc (hallucinated)

    hallucination_rate(examples: list[dict]) -> float:
      - Each example: {answer, context_docs}
      - Returns: count(examples with >= 1 hallucinated rate) / total
      - Target: < 0.03

    citation_coverage(examples: list[dict]) -> float:
      - Each example: {answer, context_docs}
      - Coverage = fraction of examples where answer contains at least one verbatim
        phrase (>= 5 words) from a context doc
      - Use sliding window: check all 5-word ngrams from answer against context
      - Target: > 0.90

    compute_all_metrics(examples: list[dict]) -> dict:
      - Returns {"hallucination_rate": float, "citation_coverage": float}
      - Prints colored pass/fail against targets using Rich

    Each example dict: {question, answer, context_docs: list[str], ground_truth: str (optional)}
  </action>
  <verify>python -c "
from src.evaluation.metrics import hallucination_rate, citation_coverage
examples = [{'answer': 'The rate is 3.5%', 'context_docs': ['The rate is 3.5% for FHA loans']}]
assert hallucination_rate(examples) == 0.0
examples2 = [{'answer': 'The rate is 7.9%', 'context_docs': ['The rate is 3.5% for FHA loans']}]
assert hallucination_rate(examples2) == 1.0
print('metrics OK')
"</verify>
  <done>Both assertions pass — correctly identifies 3.5% as grounded, 7.9% as hallucinated</done>
</task>
