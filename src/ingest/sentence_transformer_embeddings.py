import asyncio
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from llama_index.core.base.embeddings.base import BaseEmbedding


class SentenceTransformerEmbedding(BaseEmbedding):
    """Llama-Index compatible embedding wrapper using sentence-transformers.

    Implements the minimal sync + async methods used by Llama-Index's
    `BaseEmbedding` so it can be passed as `embed_model=` to indices.

    Args:
        model_name: name of the sentence-transformers model to load.
        embed_batch_size: batch size used by `get_text_embedding_batch`.
        kwargs: forwarded to BaseEmbedding.
    """

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        embed_batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        # load model (this will download weights on first run)
        model = SentenceTransformer(model_name)
        dims = model.get_sentence_embedding_dimension()

        # initialize BaseEmbedding with dimensions and batch size
        super().__init__(
            embed_batch_size=embed_batch_size,
            dimensions=dims,
            model_name=model_name,
            **kwargs,
        )

        # assign model after BaseEmbedding init to avoid pydantic attribute errors
        self._model = model
        # keep model_name attribute available
        object.__setattr__(self, "model_name", model_name)

    @classmethod
    def class_name(cls) -> str:
        return "SentenceTransformerEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> List[float]:
        emb = self._model.encode(text, convert_to_numpy=True)
        return emb.tolist()

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # sentence-transformers supports batched encoding
        embs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [e.tolist() for e in embs]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self._get_text_embeddings, texts)
