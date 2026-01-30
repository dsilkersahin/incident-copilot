from typing import Any, Dict, Generator, List, Sequence

from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)

try:
    from transformers import pipeline
    import torch
except Exception:  # pragma: no cover - optional heavy deps
    pipeline = None
    torch = None


class HuggingFaceLLM(LLM):
    """Lightweight HuggingFace Transformers LLM adapter for Llama-Index.

    This implements the minimal sync interfaces used by Llama-Index. It uses
    the `transformers.pipeline` API for text generation. Streaming is emulated
    by yielding the full generation as a single delta (not tokenwise).
    """

    model_name: str

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        device: int | None = None,
        generate_kwargs: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if pipeline is None:
            raise ImportError(
                "transformers and torch are required to use HuggingFaceLLM."
            )

        # determine device: -1 for CPU, 0..n for CUDA
        if device is None:
            if torch is not None and torch.cuda.is_available():
                device = 0
            else:
                device = -1

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.generate_kwargs = generate_kwargs or {}

        # choose pipeline task: prefer text2text for instruction-style models
        task = "text2text-generation"
        try:
            self._pipe = pipeline(
                task,
                model=self.model_name,
                device=device,
            )
        except Exception:
            # fallback to text-generation
            task = "text-generation"
            self._pipe = pipeline(task, model=self.model_name, device=device)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=2048,
            num_output=self.max_new_tokens,
            is_chat_model=False,
            model_name=self.model_name,
        )

    # -- Completion API --
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            **self.generate_kwargs,
            **kwargs,
        )
        outputs = self._pipe(prompt, **gen_kwargs)
        # transformers pipelines return a list of generations
        text = outputs[0].get("generated_text", outputs[0].get("text", ""))
        return CompletionResponse(text=text, raw=outputs[0])

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        # Not true token streaming; yield the full completion once.
        resp = self.complete(prompt, formatted=formatted, **kwargs)
        yield CompletionResponse(text=resp.text, raw=resp.raw, delta=resp.text)

    # -- Chat API --
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Convert chat messages to a single prompt string
        prompt = "\n".join([m.content if isinstance(m.content, str) else str(m.content) for m in messages])
        comp = self.complete(prompt, **kwargs)
        return ChatResponse(message=ChatMessage(role="assistant", content=comp.text), raw=comp.raw, delta=comp.text)

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        comp = self.chat(messages, **kwargs)
        yield ChatResponse(message=comp.message, raw=comp.raw, delta=comp.delta)

    # -- Async (run sync impl in thread) --
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        from llama_index.core.async_utils import asyncio_run

        return await asyncio_run(lambda: self.complete(prompt, formatted=formatted, **kwargs))

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        from llama_index.core.async_utils import asyncio_run

        def gen():
            for r in self.stream_complete(prompt, formatted=formatted, **kwargs):
                yield r

        return await asyncio_run(lambda: gen())

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        from llama_index.core.async_utils import asyncio_run

        return await asyncio_run(lambda: self.chat(messages, **kwargs))

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        from llama_index.core.async_utils import asyncio_run

        def gen():
            for r in self.stream_chat(messages, **kwargs):
                yield r

        return await asyncio_run(lambda: gen())
