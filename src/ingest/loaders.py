from llama_index.core import SimpleDirectoryReader

def load_documents(path: str):
    reader = SimpleDirectoryReader(path, recursive=True)
    return reader.load_data()
