from llama_index.core.node_parser import SentenceSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def get_splitter():
    return SentenceSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
