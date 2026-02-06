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

## Model Options

This project defaults to using the HuggingFace model `google/flan-t5-large` for generation in local experiments. Below are quick notes on that model and alternative options you can configure.

- **HuggingFace (local / on-host):**
  - `google/flan-t5-large`: A strong instruction-tuned encoder-decoder model (good balance of accuracy vs resource use). Works well for instruction-following tasks and summarization.
  - Lighter options: `google/flan-t5-base`, `google/flan-t5-small` (faster, smaller memory footprint).
  - Larger options: `google/flan-t5-xl` (higher quality, more memory/compute required).
  - Other open-source families: `EleutherAI/gpt-j-6B`, `bigscience/bloom`, or community chat models (`meta-llama/Llama-2-7b-chat-hf`) — choose based on license and hardware capability.

- **Using a HuggingFace model in code:** set the model name in your config or invocation, for example in Python:

```python
model_name = "google/flan-t5-large"
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# generate on CPU/GPU as available; consider quantization for large models
```

- **Device / performance tips:**
  - Use GPU where available (set `device_map` or move tensors to CUDA).
  - For very large models, consider 8-bit/4-bit quantization (packages like `bitsandbytes`) or model sharding.

- **OpenAI / external hosted models:**
  - If you prefer hosted APIs, you can use OpenAI models (e.g., `gpt-4o`, `gpt-3.5-turbo`) by switching the generation backend to call the OpenAI API and providing `OPENAI_API_KEY` in your environment.
  - Example (pseudo-configuration):

```text
# use_openai=true
# openai_model=gpt-4o
# set OPENAI_API_KEY in environment
```

- **Choosing a model:** balance cost, latency, and accuracy. Local HF models remove API costs but require suitable hardware and storage. Hosted models reduce infra overhead but add latency and cost.

If you want, I can add a small example config file (e.g., `config.example.yml`) showing how to switch between `google/flan-t5-large` and an OpenAI model and wire that into the code paths.




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
```

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


 


