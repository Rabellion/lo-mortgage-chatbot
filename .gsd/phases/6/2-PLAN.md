---
phase: 6
plan: 2
wave: 2
---

# Plan 6.2: FastAPI App + Docker Compose

## Objective
Wire the inference engine into FastAPI with REST and SSE streaming endpoints, lead capture, and Dockerize the full stack. This is the final deliverable before handoff.

## Context
- .gsd/SPEC.md
- .gsd/ARCHITECTURE.md
- src/inference/engine.py
- src/inference/guardrails.py
- src/config.py

## Tasks

<task type="auto">
  <name>Write FastAPI app + routes</name>
  <files>
    src/api/__init__.py
    src/api/main.py
    src/api/routes/__init__.py
    src/api/routes/chat.py
    src/api/routes/leads.py
  </files>
  <action>
    Write src/api/main.py:
      - FastAPI app with CORSMiddleware (allow all origins for MVP)
      - Include chat router at /api/chat
      - Include leads router at /api/leads
      - GET /health → {"status": "ok", "model": settings.model_name}

    Write src/api/routes/chat.py:
      POST /api/chat/:
        Body: {question: str, session_id: str (optional)}
        - check_query guardrail → 400 if off-topic with message
        - generate(question, stream=False)
        - apply_guardrails on response
        - Return: {answer: str, session_id: str}

      GET /api/chat/stream:
        Query param: question: str
        - check_query guardrail → return SSE error event if off-topic
        - StreamingResponse with media_type="text/event-stream"
        - generate(question, stream=True) → yield "data: {chunk}\n\n"
        - Final event: "data: [DONE]\n\n"

    Write src/api/routes/leads.py:
      POST /api/leads/:
        Body: {name: str, email: str, phone: str (optional), question: str (optional)}
        - Validate email format (basic regex, no external library)
        - Store in module-level list (MVP in-memory — noted as future HubSpot)
        - Return: {id: str (uuid4), message: "Thank you, we'll be in touch"}

    DO NOT add authentication — MVP only.
    DO NOT add a database for leads — in-memory list explicitly noted as temporary.
  </action>
  <verify>python -c "from src.api.main import app; print('FastAPI app importable OK')"</verify>
  <done>Imports without error</done>
</task>

<task type="auto">
  <name>Write Dockerfile + docker-compose.yml</name>
  <files>
    Dockerfile
    docker-compose.yml
    pyproject.toml
  </files>
  <action>
    Write Dockerfile:
      FROM python:3.12-slim
      WORKDIR /app
      COPY pyproject.toml .
      RUN pip install --no-cache-dir -e ".[api]" (core deps only, not training)
      COPY src/ ./src/
      EXPOSE 8080
      CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

    Write docker-compose.yml:
      services:
        api:
          build: .
          ports: ["8080:8080"]
          env_file: .env
          volumes:
            - ./data:/app/data
            - ./chroma_db:/app/chroma_db
          depends_on: [redis]
        redis:
          image: redis:7-alpine
          ports: ["6379:6379"]

    Update pyproject.toml to add [project.optional-dependencies]:
      api = [core deps list]
      training = [torch, unsloth, trl, ...]
      eval = [ragas, deepeval]
      dev = [pytest, pytest-asyncio, ruff]

    DO NOT include training deps in the Docker image — too large.
  </action>
  <verify>python -c "import tomllib; open('pyproject.toml','rb') as f: tomllib.load(f); print('pyproject.toml valid TOML')"</verify>
  <done>pyproject.toml parses as valid TOML without error</done>
</task>
