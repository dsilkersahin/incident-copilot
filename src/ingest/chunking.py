from llama_index.core.node_parser import SentenceSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def get_splitter():
    return SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
