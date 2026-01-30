from typing import Any, Generator, Sequence

import os
from transformers import pipeline, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata


class HFLocalLLM(CustomLLM):
    """A tiny HuggingFace-backed LLM adapter for local inference.

    This implements the minimal `complete`/`stream_complete` interface used by
    Llama-Index via `CustomLLM`.
    """

    def __init__(self, model_name: str = "google/flan-t5-small", pipeline_kwargs: dict | None = None, **kwargs: Any):
        # Avoid assigning pydantic-managed attrs before super().__init__.
        _model_name = model_name
        _pipeline_kwargs = pipeline_kwargs or {}

        super().__init__(**kwargs)

        # set private attrs after pydantic initialization
        object.__setattr__(self, "_model_name", _model_name)
        object.__setattr__(self, "_pipeline_kwargs", _pipeline_kwargs)
        # Create the transformers pipeline lazily to avoid heavy work during import
        # if the object is created but not used.
        object.__setattr__(self, "_pipe", None)

    @classmethod
    def class_name(cls) -> str:
        return "hf_local"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=2048,
            num_output=512,
            is_chat_model=False,
            model_name=self._model_name,
        )

    def _ensure_pipe(self) -> None:
        if self._pipe is None:
            # detect encoder-decoder models and choose appropriate pipeline task
            use_auth = os.environ.get("HF_TOKEN")
            try:
                config = AutoConfig.from_pretrained(self._model_name, use_auth_token=use_auth)
                is_enc_dec = getattr(config, "is_encoder_decoder", False)
            except Exception:
                is_enc_dec = False

            task = self._pipeline_kwargs.pop("task", None)
            if task is None:
                # prefer text-generation for causal models; for seq2seq we'll try pipeline
                task = "text-generation" if not is_enc_dec else "text2text-generation"

            # pass HF token if present
            pipe_kwargs = dict(self._pipeline_kwargs)
            if use_auth:
                pipe_kwargs.setdefault("use_auth_token", use_auth)

            try:
                # try to create a normal pipeline (may raise KeyError if task unavailable)
                self._pipe = pipeline(task, model=self._model_name, **pipe_kwargs)
            except KeyError:
                # fallback for encoder-decoder models when the pipeline task isn't registered:
                # load tokenizer + seq2seq model and create a small wrapper that mimics pipeline output
                if is_enc_dec:
                    # don't pass use_auth_token to from_pretrained for compatibility
                    tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                    model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
                    device = 0 if torch.cuda.is_available() else -1
                    if device == 0:
                        model.to("cuda")

                    def seq2seq_wrapper(prompt: str, **gen_kwargs):
                        # prepare inputs
                        inputs = tokenizer(prompt, return_tensors="pt")
                        if device == 0:
                            inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        # prefer max_new_tokens
                        gen_kwargs.pop("max_length", None)
                        max_new_tokens = gen_kwargs.pop("max_new_tokens", None)
                        if max_new_tokens is not None:
                            gen_kwargs.setdefault("max_new_tokens", max_new_tokens)
                        out_ids = model.generate(**inputs, **gen_kwargs)
                        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                        return [{"generated_text": text}]

                    self._pipe = seq2seq_wrapper
                else:
                    # re-raise if causal pipeline missing
                    raise

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        self._ensure_pipe()
        gen_kwargs = {}
        # allow overriding generation kwargs
        gen_kwargs.update(kwargs.get("gen_kwargs", {}))
        # prefer max_new_tokens and avoid conflicting max_length
        gen_kwargs.pop("max_length", None)

        outputs = self._pipe(prompt, **gen_kwargs)
        # pipeline returns a list of dicts
        text = ""
        if isinstance(outputs, list) and len(outputs) > 0:
            out0 = outputs[0]
            text = out0.get("generated_text") or out0.get("text") or out0.get("summary_text") or ""
        else:
            # fallback to stringifying the output
            text = str(outputs)
        return CompletionResponse(text=text, additional_kwargs={})

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        # For simplicity, return the full text as a single chunk (no token-level streaming).
        resp = self.complete(prompt, formatted=formatted, **kwargs)
        yield resp
