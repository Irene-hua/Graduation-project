"""LLM abstraction layer.

This project runs on Windows + CPU-only and must use Ollama (GGUF/llama.cpp) for
quantized inference. To keep the RAG pipeline decoupled from the backend, we
define a tiny interface that any LLM implementation can follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMResult:
    """Normalized LLM response.

    Attributes:
        text: The generated answer text.
        model: Model identifier (e.g., 'mistral', 'llama2').
        raw: Optional backend raw payload for debugging.
        inference_time: Optional backend timing (seconds).
    """

    text: str
    model: str
    raw: Optional[Dict[str, Any]] = None
    inference_time: Optional[float] = None


class BaseLLM(ABC):
    """Minimal interface used by the RAG pipeline."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResult:
        raise NotImplementedError

    def is_available(self) -> bool:
        """Whether the LLM backend is reachable/ready."""

        return True

