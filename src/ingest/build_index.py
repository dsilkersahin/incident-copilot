from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from src.ingest.sentence_transformer_embeddings import (
    SentenceTransformerEmbedding,
)
from src.ingest.loaders import load_documents
from src.ingest.chunking import get_splitter
from src.config import INDEX_PATH

import faiss
import os

def build_index(data_dir="data/raw"):
    docs = load_documents(data_dir)

    splitter = get_splitter()
    nodes = splitter.get_nodes_from_documents(docs)

    # determine embedding model and its embedding dimension
    embed_model = SentenceTransformerEmbedding()
    dimension = getattr(embed_model, "dimensions", None)
    if dimension is None:
        # fallback: ask underlying model for dimension if available
        try:
            dimension = embed_model._model.get_sentence_embedding_dimension()
        except Exception:
            # default to a common dimension (mpnet = 768)
            dimension = 768

    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    os.makedirs(INDEX_PATH, exist_ok=True)
    index.storage_context.persist(persist_dir=INDEX_PATH)

    print("Index built and saved.")

if __name__ == "__main__":
    build_index()
