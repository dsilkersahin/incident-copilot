from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from src.ingest.loaders import load_documents
from src.ingest.chunking import get_splitter
from src.config import INDEX_PATH

import faiss
import os

def build_index(data_dir="data/raw"):
    docs = load_documents(data_dir)

    splitter = get_splitter()
    nodes = splitter.get_nodes_from_documents(docs)

    dimension = 3072
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=OpenAIEmbedding()
    )

    os.makedirs(INDEX_PATH, exist_ok=True)
    index.storage_context.persist(persist_dir=INDEX_PATH)

    print("Index built and saved.")

if __name__ == "__main__":
    build_index()
