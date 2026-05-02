"""LLM module for local language model deployment"""

from .base import BaseLLM, LLMResult
from .ollama import OllamaClient, OllamaLLM

__all__ = ['BaseLLM', 'LLMResult', 'OllamaClient', 'OllamaLLM']
