"""Ollama-backed LLM implementation.

This project runs on Windows + CPU-only and uses Ollama (GGUF/llama.cpp) for
quantized inference. This module provides a small adapter that conforms to the
project's `BaseLLM` interface while delegating HTTP calls to `OllamaClient`.

Supported models include (example): 'llama2', 'mistral'.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import BaseLLM, LLMResult
from .ollama_client import OllamaClient


class OllamaLLM(BaseLLM):
    """BaseLLM implementation backed by Ollama."""

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
    ) -> None:
        self.model_name = model_name
        self._client = OllamaClient(base_url=base_url, model_name=model_name, timeout=timeout)

    def is_available(self) -> bool:
        return self._client.is_available()

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResult:
        data = self._client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return LLMResult(
            text=(data.get("response") or ""),
            model=str(data.get("model") or self.model_name),
            raw=data.get("raw"),
            inference_time=data.get("inference_time"),
        )

