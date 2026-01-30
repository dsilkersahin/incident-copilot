import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
INDEX_PATH = os.getenv("INDEX_PATH", "indexes/faiss")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5
