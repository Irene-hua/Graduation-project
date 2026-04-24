# ...existing code...
"""Ollama client + adapter merged module.

This module provides both the low-level HTTP client (`OllamaClient`) and the
`OllamaLLM` adapter that implements the project's `BaseLLM` interface.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from .base import BaseLLM, LLMResult

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for Ollama API"""

    def __init__(self,
                 base_url: str = 'http://localhost:11434',
                 model_name: str = 'llama2',
                 timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        logger.info(f"Initialized Ollama client for model: {model_name}")

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama server not available: {e}")
            return False

    def list_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def generate(self,
                 prompt: str,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 stream: bool = False,
                 **kwargs) -> Dict:
        url = f"{self.base_url}/api/generate"

        payload = {
            'model': self.model_name,
            'prompt': prompt,
            'stream': stream,
            'options': {
                'temperature': temperature,
            }
        }

        if max_tokens:
            payload['options']['num_predict'] = max_tokens

        payload['options'].update(kwargs)

        start_time = time.time()

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            inference_time = time.time() - start_time
            resp_text = data.get('response') or data.get('output') or ''

            return {
                'response': resp_text,
                'model': self.model_name,
                'inference_time': inference_time,
                'eval_count': data.get('eval_count', 0),
                'eval_duration': data.get('eval_duration', 0),
                'raw': data
            }

        except requests.exceptions.Timeout:
            logger.error(f"Request timeout after {self.timeout}s")
            raise
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def chat(self,
             messages: List[Dict[str, str]],
             temperature: float = 0.7,
             max_tokens: Optional[int] = None) -> Dict:
        url = f"{self.base_url}/api/chat"

        payload = {
            'model': self.model_name,
            'messages': messages,
            'stream': False,
            'options': {
                'temperature': temperature,
            }
        }

        if max_tokens:
            payload['options']['num_predict'] = max_tokens

        start_time = time.time()

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            inference_time = time.time() - start_time

            return {
                'response': data.get('message', {}).get('content', ''),
                'model': self.model_name,
                'inference_time': inference_time
            }

        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise

    def get_model_info(self) -> Dict:
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={'name': self.model_name},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}


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


__all__ = ["OllamaClient", "OllamaLLM"]

