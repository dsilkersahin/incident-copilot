# Incident Copilot (LLM + RAG Baseline)

An experimental incident-response assistant built using **Retrieval-Augmented Generation (RAG)**.

The system retrieves relevant context from runbooks and postmortems using vector search and generates answers using a local HuggingFace LLM.

---

## What This Is
- Pure **LLM + RAG** implementation
- FAISS + Sentence Transformers for retrieval
- LlamaIndex for orchestration
- Local Flan-T5 model for generation

---

## What It Can Do
- Answer incident-related questions
- Summarize runbooks and postmortems
- Attempt step-by-step answers for “how” questions

---

## What It Is Not
- No deterministic step extraction
- No rule-based safeguards
- No guarantees on procedural completeness

This repository represents a **baseline RAG approach** and is intended for experimentation and learning.

---



# 1) setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) put docs under data/raw/
.md, pdf, txt etc.

# 3) build index
python -m src.ingest.build_index

# 4) run API
uvicorn src.api.main:app --reload

# 5) ask
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How do I restart Service X?"}'

## Example

```bash
python src/generation/answer.py \
  --question "How do I restart Service X?"

# 6) Docker Runbook

Quick commands to build, run, and inspect the project using Docker / Docker Compose.

- **Prereqs:** Docker and Docker Compose (v2) installed locally.

- **Build images:**

```bash
docker compose build
```

- **Start services (foreground):**

```bash
docker compose up
```

- **Start services (detached):**

```bash
docker compose up -d
```

- **View logs (all services):**

```bash
docker compose logs -f
```

- **View logs (single service):**

```bash
docker compose logs -f <service-name>
```

- **Rebuild and restart (useful after code changes):**

```bash
docker compose up -d --build
```

- **Run a shell inside the web container:**

```bash
docker compose exec <service-name> /bin/bash
# or
docker compose exec <service-name> /bin/sh
```

- **Stop and remove containers:**

```bash
docker compose down
```

- **Persisted data / volumes:** Check `docker-compose.yml` for mounted volumes (database, indexes, or data directories). Back up `data/processed/` and `indexes/` before destructive operations.