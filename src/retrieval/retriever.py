from llama_index.core import StorageContext, load_index_from_storage
from src.config import INDEX_PATH, TOP_K

def get_retriever():
    storage = StorageContext.from_defaults(persist_dir=INDEX_PATH)
    index = load_index_from_storage(storage)
    return index.as_retriever(similarity_top_k=TOP_K)

def retrieve(question: str):
    retriever = get_retriever()
    return retriever.retrieve(question)
