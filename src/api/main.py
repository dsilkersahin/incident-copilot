from fastapi import FastAPI
from pydantic import BaseModel
from src.generation.answer import ask

app = FastAPI(title="Incident Copilot")

class AskRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask_question(req: AskRequest):
    return ask(req.question)
