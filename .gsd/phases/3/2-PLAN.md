---
phase: 3
plan: 2
wave: 2
---

# Plan 3.2: Triplet Validation + ShareGPT Conversion

## Objective
Validate generated triplets for quality (CoT citations, no fabricated rates) and convert valid ones to ShareGPT format for Unsloth. This determines the actual training set quality.

## Context
- .gsd/SPEC.md
- .gsd/ARCHITECTURE.md
- src/raft/synthesis.py
- data/raft_triplets/raw_triplets.jsonl (produced by synthesis)

## Tasks

<task type="auto">
  <name>Write validation module</name>
  <files>
    src/raft/validation.py
  </files>
  <action>
    Write src/raft/validation.py with:

    validate_triplet(triplet: dict) -> tuple[bool, list[str]]:
      Check all of:
      - question exists and len >= 10
      - oracle_doc exists and len >= 20
      - answer exists and len >= 10
      - chain_of_thought exists and contains "##begin_quote##" AND "##end_quote##"
      - distractor_docs list has >= 1 entry
      - Rate fabrication check: find all "\d+\.?\d*\s*%" in answer;
        for each rate, verify it appears verbatim in oracle_doc.
        If any rate NOT in oracle: flag as fabricated.
      Return (True, []) if all pass, (False, [issue, ...]) otherwise

    validate_dataset(input_path, valid_output_path=None, invalid_output_path=None) -> dict:
      - Stream through JSONL line by line (do not load all into RAM)
      - Write valid/invalid to separate JSONL files
      - Return stats: {total, valid, invalid, issues: {issue_text: count}}

    Issue strings must be consistent (used for aggregation in the script).
  </action>
  <verify>python -c "from src.raft.validation import validate_triplet; ok, issues = validate_triplet({'question': 'test question here', 'oracle_doc': 'doc content here', 'answer': 'answer here', 'chain_of_thought': '##begin_quote## text ##end_quote##', 'distractor_docs': ['d1']}); assert ok, issues"</verify>
  <done>Assert passes — valid triplet is accepted</done>
</task>

<task type="auto">
  <name>Write ShareGPT conversion module + generate script</name>
  <files>
    src/raft/dataset.py
    scripts/generate_raft_data.py
  </files>
  <action>
    Write src/raft/dataset.py with:

    SYSTEM_MESSAGE: "You are a knowledgeable and friendly mortgage advisor. Answer the client's
      question using ONLY the provided documents. Cite the relevant document when answering.
      If the documents don't contain the answer, say so honestly. Never fabricate rates,
      figures, or eligibility criteria. Speak in warm, jargon-light conversational English."

    triplet_to_sharegpt(triplet) -> dict:
      - Combine oracle_doc + distractor_docs, shuffle randomly
      - Format as <DOCUMENT id=N> blocks
      - conversations: [system, human (docs + question), gpt (CoT + answer)]

    triplet_to_alpaca(triplet) -> dict:
      - instruction=question, input=doc context, output=CoT+answer

    convert_dataset(input_path, output_path, fmt="sharegpt") -> int:
      - Stream JSONL input, convert each, write to JSON array output
      - Return count

    ---

    Write scripts/generate_raft_data.py with:
    - --target default=20000, --questions-per-chunk default=5, --concurrency default=30
    - Pull chunks from ChromaDB collection
    - Calculate chunks_needed = ceil(target * 1.2 / questions_per_chunk) for 20% buffer
    - Generate with Rich progress bar (SpinnerColumn + BarColumn + TimeElapsedColumn)
    - Validate → convert → print final stats
    - Warn clearly if valid count < target (user needs more source docs or lower target)
  </action>
  <verify>python -c "from src.raft.dataset import triplet_to_sharegpt; r = triplet_to_sharegpt({'question':'q','oracle_doc':'o','distractor_docs':['d'],'chain_of_thought':'##begin_quote## x ##end_quote##','answer':'a'}); assert len(r['conversations'])==3"</verify>
  <done>Assert passes — ShareGPT dict has 3 conversation turns (system, human, gpt)</done>
</task>
