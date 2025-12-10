"""
Quantized Model
Support for 4-bit quantization using QLoRA/bitsandbytes
"""

import torch
from typing import Optional, Dict
import logging
import time
import psutil
import os

logger = logging.getLogger(__name__)


class QuantizedModel:
    """Wrapper for quantized language models using bitsandbytes"""
    
    def __init__(self, 
                 model_name: str,
                 quantization_bits: int = 4,
                 quantization_type: str = 'nf4',
                 device_map: str = 'auto'):
        """
        Initialize quantized model
        
        Args:
            model_name: HuggingFace model name
            quantization_bits: Bits for quantization (4 or 8)
            quantization_type: Type of quantization ('nf4' or 'fp4')
            device_map: Device mapping strategy
        """
        self.model_name = model_name
        self.quantization_bits = quantization_bits
        self.quantization_type = quantization_type
        
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        logger.info(f"Initialized quantized model config: {model_name} ({quantization_bits}-bit {quantization_type})")
    
    def load_model(self):
        """Load model with quantization"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            logger.info(f"Loading model with {self.quantization_bits}-bit quantization...")
            
            # Configure quantization
            if self.quantization_bits == 4:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.quantization_type,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            elif self.quantization_bits == 8:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                raise ValueError(f"Unsupported quantization bits: {self.quantization_bits}")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map='auto',
                trust_remote_code=True
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.is_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(self, 
                prompt: str,
                max_new_tokens: int = 256,
                temperature: float = 0.7,
                top_p: float = 0.9,
                do_sample: bool = True) -> Dict:
        """
        Generate text with the quantized model
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Dict with generated text and metadata
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Get initial memory usage
        memory_before = self._get_memory_usage()
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        start_time = time.time()
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        inference_time = time.time() - start_time
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        # Get final memory usage
        memory_after = self._get_memory_usage()
        
        return {
            'response': generated_text,
            'inference_time': inference_time,
            'tokens_generated': len(outputs[0]) - len(inputs['input_ids'][0]),
            'memory_usage_mb': memory_after['ram_mb'],
            'memory_increase_mb': memory_after['ram_mb'] - memory_before['ram_mb'],
            'gpu_memory_mb': memory_after.get('gpu_mb', 0)
        }
    
    def _get_memory_usage(self) -> Dict:
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / 1024 / 1024
        
        result = {'ram_mb': ram_mb}
        
        # Get GPU memory if available
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
            result['gpu_mb'] = gpu_mb
        
        return result
    
    def get_model_size(self) -> Dict:
        """Get model size information"""
        if not self.is_loaded:
            return {}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate size
        param_size_mb = total_params * self.quantization_bits / 8 / 1024 / 1024
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'quantization_bits': self.quantization_bits,
            'estimated_size_mb': param_size_mb
        }
    
    def benchmark(self, 
                 prompts: list,
                 max_new_tokens: int = 256) -> Dict:
        """
        Benchmark model performance
        
        Args:
            prompts: List of test prompts
            max_new_tokens: Maximum tokens per generation
            
        Returns:
            Dict with benchmark results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        inference_times = []
        token_counts = []
        
        for prompt in prompts:
            result = self.generate(prompt, max_new_tokens=max_new_tokens)
            inference_times.append(result['inference_time'])
            token_counts.append(result['tokens_generated'])
        
        import numpy as np
        
        return {
            'num_prompts': len(prompts),
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'avg_tokens_per_second': np.mean([t / time for t, time in zip(token_counts, inference_times)]),
            'model_info': self.get_model_size()
        }
