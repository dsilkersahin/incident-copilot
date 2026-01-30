from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from src.config import INDEX_PATH, MODEL_NAME
from src.generation.prompts import SYSTEM_PROMPT

def ask(question: str):
    storage = StorageContext.from_defaults(persist_dir=INDEX_PATH)
    index = load_index_from_storage(storage)

    engine = index.as_query_engine(
        llm=OpenAI(model=MODEL_NAME),
        system_prompt=SYSTEM_PROMPT
    )

    response = engine.query(question)

    return {
        "answer": str(response),
        "sources": [
            node.node.metadata.get("file_name", "unknown")
            for node in response.source_nodes
        ]
    }
