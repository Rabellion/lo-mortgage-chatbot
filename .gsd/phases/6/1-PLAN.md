---
phase: 6
plan: 1
wave: 1
---

# Plan 6.1: RAG Inference Engine + Guardrails

## Objective
Build the inference layer: ChromaDB retriever, RAG engine, and multi-layer guardrails. This is the ML contribution to the API — the API team wires it into FastAPI, but this module must work standalone first.

## Context
- .gsd/SPEC.md
- .gsd/ARCHITECTURE.md
- src/config.py
- src/data/embedding.py

## Tasks

<task type="auto">
  <name>Write retriever and inference engine</name>
  <files>
    src/inference/retriever.py
    src/inference/engine.py
    src/inference/__init__.py
  </files>
  <action>
    Write src/inference/retriever.py:
      retrieve(query: str, collection_name: str = "mortgage_docs", top_k: int = None) -> list[dict]:
        - top_k defaults to settings.retriever_top_k (5)
        - Returns list of {text, metadata, score} dicts
        - Score = 1 - cosine distance (similarity, not distance)

    Write src/inference/engine.py:
      RAG_SYSTEM_PROMPT: same as training SYSTEM_MESSAGE (must match exactly)

      build_context(docs: list[dict]) -> str:
        - Format as <DOCUMENT id=N>\n{text}\n</DOCUMENT> blocks

      generate(question: str, stream: bool = False) -> str | Generator:
        - Retrieve top-5 docs via retriever
        - Build context
        - Call OpenAI-compatible API at settings.vllm_base_url with settings.model_name
        - stream=False: return full string
        - stream=True: return generator yielding text chunks (for SSE)

      DO NOT hardcode the model name or URL — use settings throughout.
      DO NOT handle guardrails here — guardrails are applied by the caller.
  </action>
  <verify>python -c "from src.inference.engine import build_context; ctx = build_context([{'text': 'hello', 'metadata': {}, 'score': 0.9}]); assert 'DOCUMENT id=1' in ctx; print('engine OK')"</verify>
  <done>Assert passes — context formatted with DOCUMENT tags</done>
</task>

<task type="auto">
  <name>Write guardrails module</name>
  <files>
    src/inference/guardrails.py
  </files>
  <action>
    Write src/inference/guardrails.py with:

    OFF_TOPIC_KEYWORDS: list of strings indicating off-topic queries:
      ["stock", "crypto", "bitcoin", "gambling", "forex", "options trading",
       "commodity", "hedge fund", "insurance policy", "tax return"]

    DISCLAIMER = "This information is for educational purposes only and does not
      constitute financial or legal advice. Please consult a licensed mortgage
      professional for guidance specific to your situation."

    check_query(query: str) -> tuple[bool, str | None]:
      - Returns (True, None) if query is on-topic
      - Returns (False, "I can only help with mortgage and loan questions.") if off-topic
      - Off-topic detection: any OFF_TOPIC_KEYWORD appears in query.lower()

    check_response(response: str, context_docs: list[str]) -> str:
      - Detect fabricated rates: any "X.X%" in response not in any context_doc
      - If found: replace the specific rate with "[rate not available — please consult a lender]"
      - Detect over-promising language: "you will qualify", "guaranteed", "you are approved"
        → replace with "you may qualify", "likely", "you may be approved"
      - Always append DISCLAIMER (separated by \n\n) if not already present

    apply_guardrails(query: str, response: str, context_docs: list[str]) -> tuple[bool, str]:
      - Run check_query first; if off-topic return (False, off_topic_message)
      - Run check_response on response
      - Return (True, clean_response)
  </action>
  <verify>python -c "
from src.inference.guardrails import check_query, check_response
ok, _ = check_query('What mortgage rate can I get?')
assert ok
ok2, msg = check_query('Should I invest in bitcoin?')
assert not ok2
resp = check_response('The rate is 7.9%.', ['The current rate is 3.5%.'])
assert '7.9%' not in resp or 'not available' in resp
print('guardrails OK')
"</verify>
  <done>All 3 assertions pass</done>
</task>
