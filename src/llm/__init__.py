"""LLM module for local language model deployment"""
from .ollama_client import OllamaClient
from .quantized_model import QuantizedModel

__all__ = ['OllamaClient', 'QuantizedModel']
