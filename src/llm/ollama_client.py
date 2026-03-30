"""
Ollama Client
Interface for local LLM deployment using Ollama
"""

import requests
import json
from typing import Optional, Dict, List
import logging
import time

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for Ollama API"""

    def __init__(self,
                 base_url: str = 'http://localhost:11434',
                 model_name: str = 'llama2',
                 timeout: int = 300):
        """
        Initialize Ollama client

        Args:
            base_url: Base URL for Ollama API
            model_name: Name of the model to use
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout

        logger.info(f"Initialized Ollama client for model: {model_name}")

    def is_available(self) -> bool:
        """Check if Ollama server is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama server not available: {e}")
            return False

    def list_models(self) -> List[str]:
        """List available models"""
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
        """
        Generate text using Ollama

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            **kwargs: Additional generation parameters

        Returns:
            Dict with 'response' and metadata
        """
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

        # Add any additional options
        payload['options'].update(kwargs)

        start_time = time.time()

        try:
            # Use non-streaming default for robustness; streaming parsing varies by Ollama version.
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            inference_time = time.time() - start_time

            # Ollama may return {'response': '...'} or {'output': '...'} depending on version; be flexible.
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
        """
        Chat completion using Ollama

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dict with response
        """
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
        """Get information about the current model"""
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
