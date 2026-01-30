from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import os

from src.config import INDEX_PATH, MODEL_NAME
from src.generation.prompts import SYSTEM_PROMPT
from src.generation.hf_local_llm import HFLocalLLM
from src.ingest.sentence_transformer_embeddings import SentenceTransformerEmbedding


def ask(question: str):
    # Some persisted FAISS files in the index folder may be raw binary
    # faiss index files (which can cause the default loader to attempt
    # to json.load a binary file). Try to load the faiss index and
    # pass it into StorageContext so the vector store is reconstructed
    # from the binary blob.
    faiss_index_path = os.path.join(INDEX_PATH, "default__vector_store.json")
    try:
        if os.path.exists(faiss_index_path):
            faiss_index = faiss.read_index(faiss_index_path)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage = StorageContext.from_defaults(persist_dir=INDEX_PATH, vector_store=vector_store)
        else:
            storage = StorageContext.from_defaults(persist_dir=INDEX_PATH)
    except Exception:
        # fallback to the default loader (may raise if persist is malformed)
        storage = StorageContext.from_defaults(persist_dir=INDEX_PATH)

    # Override the saved embed_model with our local SentenceTransformer embeddings
    embed_model_override = SentenceTransformerEmbedding()
    index = load_index_from_storage(storage, embed_model=embed_model_override)

    # If MODEL_NAME looks like an OpenAI model id (default in config),
    # fall back to a small HF model so the code runs without OpenAI.
    model_name = MODEL_NAME
    if model_name is None or model_name.lower().startswith("gpt-") or model_name.lower().startswith("gpt"):
        model_name = "google/flan-t5-small"

    engine = index.as_query_engine(
        llm=HFLocalLLM(model_name=model_name),
        system_prompt=SYSTEM_PROMPT,
    )

    response = engine.query(question)

    return {
        "answer": str(response),
        "sources": [
            node.node.metadata.get("file_name", "unknown")
            for node in response.source_nodes
        ],
    }
