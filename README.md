# Incident Copilot (Scaffold)

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill OPENAI_API_KEY later
uvicorn src.api.main:app --reload
