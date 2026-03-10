from src.config import INDEX_PATH, MODEL_NAME
from src.generation.prompts import SYSTEM_PROMPT
from src.generation.hf_local_llm import HFLocalLLM
# Note: this "LLM only" script avoids importing llama_index/faiss so it can
# run in environments where those packages are not installed. The full index
# path (ask()) still requires llama_index/faiss.


def is_procedural(question: str) -> bool:
    q = question.lower()
    return (
        q.startswith("how ")
        or "restart" in q
        or "steps" in q
        or "procedure" in q
    )

import re

def strip_headers(text: str) -> str:
    return re.sub(r"^#+\s.*$", "", text, flags=re.MULTILINE).strip()


# ...existing code...
def ask_llm_only(question: str, model_name: str | None = None):
    """
    Generate an answer using only the LLM (no index / retrieval).
    Returns same structure as ask().
    """
    if model_name is None:
        model_name = MODEL_NAME
        if model_name is None or model_name.lower().startswith("gpt-") or model_name.lower().startswith("gpt"):
            model_name = "google/flan-t5-small"

    llm = HFLocalLLM(model_name=model_name)
    prompt = SYSTEM_PROMPT + "\n\n" + question
    # Use the LLM's `complete` method (returns a CompletionResponse)
    resp = llm.complete(prompt, formatted=False)
    # Normalize to text
    try:
        text = resp.text.strip()
    except Exception:
        text = str(resp).strip()

    if not text:
        text = "No response from LLM."

    return {"answer": text, "sources": []}
# ...existing code...

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env")
    parser.add_argument("--question")
    parser.add_argument("--llm-only", action="store_true", help="Use LLM only (no index)")
    args = parser.parse_args()

    if args.llm_only:
        result = ask_llm_only(args.question)
        print(result)
    else:
        # The full retrieval+LLM flow depends on llama_index/faiss being
        # installed. This script intentionally avoids importing them so the
        # LLM-only path can run in lightweight environments. If you want to
        # run the full pipeline, install the project dependencies and run
        # `python src/generation/answer.py` or run this script after
        # installing `llama_index` and `faiss`.
        print({
            "error": "Full pipeline requires llama_index/faiss. Re-run with --llm-only for LLM-only mode, or install dependencies to use retrieval."
        })