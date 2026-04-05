---
phase: 5
plan: 2
wave: 2
---

# Plan 5.2: RAGAS Evaluation + Evaluation Script

## Objective
Wire up RAGAS for faithfulness/relevancy metrics and write the evaluation script that generates model answers, computes all metrics, and writes a results report.

## Context
- .gsd/SPEC.md (success criteria)
- .gsd/ARCHITECTURE.md
- src/evaluation/metrics.py
- data/raw/financebench.jsonl (eval set)
- models/gemma-4-31b-raft/ (fine-tuned model — must exist)

## Tasks

<task type="auto">
  <name>Write RAGAS evaluation wrapper</name>
  <files>
    src/evaluation/ragas_eval.py
  </files>
  <action>
    Write src/evaluation/ragas_eval.py with:

    run_ragas(examples: list[dict], llm=None, embeddings=None) -> dict:
      - Import from ragas: evaluate, faithfulness, answer_relevancy,
        context_precision, context_recall
      - Build ragas Dataset from examples:
          each example needs: question, answer, contexts (list[str]), ground_truth
      - Run evaluate() with all 4 metrics
      - Return dict with metric names and float scores

    RAGAS requires an LLM judge — use OpenAI gpt-4o-mini (cheaper for eval).
    If llm=None, default to ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)

    Handle ImportError gracefully: if ragas not installed, return empty dict with warning.
    RAGAS evals can be slow — print progress with Rich.
  </action>
  <verify>python -c "from src.evaluation.ragas_eval import run_ragas; print('ragas_eval importable OK')"</verify>
  <done>Imports without error (ragas may not be installed — ImportError should be caught gracefully)</done>
</task>

<task type="auto">
  <name>Write evaluate.py script</name>
  <files>
    scripts/evaluate.py
  </files>
  <action>
    Write scripts/evaluate.py that:

    1. Loads eval set from data/raw/financebench.jsonl (eval_only examples)
       OR from a custom data/evaluation/test_questions.jsonl if it exists

    2. For each eval example, generates a model answer:
       - Query ChromaDB for top-5 context docs
       - Call vLLM inference endpoint (settings.vllm_base_url) with the fine-tuned model
       - Store: {question, answer, context_docs, ground_truth}

    3. Computes all metrics:
       - Custom: hallucination_rate, citation_coverage
       - RAGAS: faithfulness, answer_relevancy, context_precision, context_recall

    4. Writes results to data/evaluation/results.json

    5. Prints a Rich table with:
       - Metric | Score | Target | Pass/Fail
       - hallucination_rate | X% | <3% | ✓/✗
       - citation_coverage | X% | >90% | ✓/✗
       - ragas_faithfulness | X | >0.85 | ✓/✗
       - ragas_answer_relevancy | X | >0.80 | ✓/✗

    6. Exits with code 1 if any must-have metric fails (CI-friendly)

    Flag --limit N: only evaluate first N examples (for quick smoke test)
    Flag --skip-ragas: skip RAGAS (faster, custom metrics only)
  </action>
  <verify>python scripts/evaluate.py --help</verify>
  <done>Help text prints with --limit and --skip-ragas flags documented</done>
</task>
