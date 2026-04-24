"""LLM module for local language model deployment"""

from .base import BaseLLM, LLMResult
from .ollama import OllamaClient, OllamaLLM

# Optional: legacy HF+bitsandbytes quantization path (not used in CPU-only Ollama setup).
try:
    from .quantized_model import QuantizedModel  # type: ignore
except Exception:  # pragma: no cover
    QuantizedModel = None  # type: ignore

__all__ = ['BaseLLM', 'LLMResult', 'OllamaClient', 'OllamaLLM', 'QuantizedModel']
