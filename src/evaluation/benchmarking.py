"""
Benchmarking Module
Benchmark RAG system performance with different configurations
"""

from typing import List, Dict, Optional
import time
import logging
from .metrics import Metrics

logger = logging.getLogger(__name__)


class Benchmarking:
    """Benchmarking utilities for RAG system"""
    
    def __init__(self, rag_system):
        """
        Initialize benchmarking
        
        Args:
            rag_system: RAGSystem instance to benchmark
        """
        self.rag_system = rag_system
        self.metrics = Metrics()
    
    def benchmark_retrieval_k_values(self,
                                    queries: List[str],
                                    k_values: List[int],
                                    ground_truth_ids: Optional[List[List[str]]] = None) -> Dict:
        """
        Benchmark retrieval with different K values
        
        Args:
            queries: List of test queries
            k_values: List of K values to test
            ground_truth_ids: Optional ground truth relevant IDs for each query
            
        Returns:
            Dict with results for each K value
        """
        results = {}
        
        for k in k_values:
            logger.info(f"Benchmarking with K={k}")
            
            retrieval_times = []
            num_results = []
            
            for i, query in enumerate(queries):
                start_time = time.time()
                retrieved = self.rag_system.retriever.retrieve(query, top_k=k)
                retrieval_time = time.time() - start_time
                
                retrieval_times.append(retrieval_time)
                num_results.append(len(retrieved))
            
            k_results = {
                'k': k,
                'avg_retrieval_time': sum(retrieval_times) / len(retrieval_times),
                'avg_results_returned': sum(num_results) / len(num_results),
                'min_retrieval_time': min(retrieval_times),
                'max_retrieval_time': max(retrieval_times)
            }
            
            # Calculate quality metrics if ground truth provided
            if ground_truth_ids and len(ground_truth_ids) == len(queries):
                precisions = []
                recalls = []
                f1_scores = []
                
                for i, query in enumerate(queries):
                    retrieved = self.rag_system.retriever.retrieve(query, top_k=k)
                    retrieved_ids = [r['metadata'].get('chunk_id', '') for r in retrieved]
                    
                    metrics = self.metrics.calculate_retrieval_metrics(
                        retrieved_ids,
                        ground_truth_ids[i],
                        k=k
                    )
                    
                    precisions.append(metrics['precision'])
                    recalls.append(metrics['recall'])
                    f1_scores.append(metrics['f1'])
                
                k_results.update({
                    'avg_precision': sum(precisions) / len(precisions),
                    'avg_recall': sum(recalls) / len(recalls),
                    'avg_f1': sum(f1_scores) / len(f1_scores)
                })
            
            results[f'k_{k}'] = k_results
        
        return results
    
    def benchmark_embedding_models(self,
                                  model_names: List[str],
                                  test_texts: List[str]) -> Dict:
        """
        Benchmark different embedding models
        
        Args:
            model_names: List of model names to test
            test_texts: List of texts to encode
            
        Returns:
            Dict with results for each model
        """
        from ..embedding import EmbeddingModel
        
        results = {}
        
        for model_name in model_names:
            logger.info(f"Benchmarking embedding model: {model_name}")
            
            try:
                # Load model
                load_start = time.time()
                model = EmbeddingModel(model_name)
                load_time = time.time() - load_start
                
                # Encode texts
                encode_start = time.time()
                embeddings = model.encode(test_texts, show_progress=False)
                encode_time = time.time() - encode_start
                
                results[model_name] = {
                    'load_time': load_time,
                    'encode_time': encode_time,
                    'avg_time_per_text': encode_time / len(test_texts),
                    'dimension': model.get_dimension(),
                    'texts_per_second': len(test_texts) / encode_time
                }
                
            except Exception as e:
                logger.error(f"Failed to benchmark {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def benchmark_quantized_vs_full(self,
                                   test_prompts: List[str],
                                   quantized_model,
                                   full_model=None) -> Dict:
        """
        Compare quantized and full-precision models
        
        Args:
            test_prompts: List of test prompts
            quantized_model: Quantized model instance
            full_model: Optional full-precision model for comparison
            
        Returns:
            Dict with comparison results
        """
        results = {'quantized': {}, 'full': {}}
        
        # Benchmark quantized model
        logger.info("Benchmarking quantized model")
        
        quantized_times = []
        quantized_memory = []
        
        for prompt in test_prompts:
            result = quantized_model.generate(prompt)
            quantized_times.append(result['inference_time'])
            quantized_memory.append(result.get('memory_usage_mb', 0))
        
        results['quantized'] = {
            'avg_inference_time': sum(quantized_times) / len(quantized_times),
            'avg_memory_mb': sum(quantized_memory) / len(quantized_memory) if quantized_memory else 0,
            'model_size': quantized_model.get_model_size()
        }
        
        # Benchmark full model if provided
        if full_model:
            logger.info("Benchmarking full-precision model")
            
            full_times = []
            full_memory = []
            
            for prompt in test_prompts:
                result = full_model.generate(prompt)
                full_times.append(result['inference_time'])
                full_memory.append(result.get('memory_usage_mb', 0))
            
            results['full'] = {
                'avg_inference_time': sum(full_times) / len(full_times),
                'avg_memory_mb': sum(full_memory) / len(full_memory) if full_memory else 0,
                'model_size': full_model.get_model_size()
            }
            
            # Calculate speedup and memory savings
            results['comparison'] = {
                'speedup': results['full']['avg_inference_time'] / results['quantized']['avg_inference_time'],
                'memory_reduction': 1 - (results['quantized']['avg_memory_mb'] / results['full']['avg_memory_mb']) if results['full']['avg_memory_mb'] > 0 else 0,
                'memory_reduction_pct': (1 - (results['quantized']['avg_memory_mb'] / results['full']['avg_memory_mb'])) * 100 if results['full']['avg_memory_mb'] > 0 else 0
            }
        
        return results
    
    def benchmark_encrypted_vs_baseline(self,
                                       queries: List[str],
                                       encrypted_rag,
                                       baseline_rag=None) -> Dict:
        """
        Compare encrypted RAG with non-encrypted baseline
        
        Args:
            queries: List of test queries
            encrypted_rag: Encrypted RAG system
            baseline_rag: Optional baseline RAG system without encryption
            
        Returns:
            Dict with comparison results
        """
        results = {'encrypted': {}, 'baseline': {}}
        
        # Benchmark encrypted system
        logger.info("Benchmarking encrypted RAG system")
        
        encrypted_times = []
        encrypted_retrieval_times = []
        encrypted_generation_times = []
        
        for query in queries:
            result = encrypted_rag.answer_question(query)
            encrypted_times.append(result['total_time'])
            encrypted_retrieval_times.append(result['retrieval_time'])
            encrypted_generation_times.append(result['generation_time'])
        
        results['encrypted'] = {
            'avg_total_time': sum(encrypted_times) / len(encrypted_times),
            'avg_retrieval_time': sum(encrypted_retrieval_times) / len(encrypted_retrieval_times),
            'avg_generation_time': sum(encrypted_generation_times) / len(encrypted_generation_times)
        }
        
        # Benchmark baseline if provided
        if baseline_rag:
            logger.info("Benchmarking baseline RAG system")
            
            baseline_times = []
            baseline_retrieval_times = []
            baseline_generation_times = []
            
            for query in queries:
                result = baseline_rag.answer_question(query)
                baseline_times.append(result['total_time'])
                baseline_retrieval_times.append(result['retrieval_time'])
                baseline_generation_times.append(result['generation_time'])
            
            results['baseline'] = {
                'avg_total_time': sum(baseline_times) / len(baseline_times),
                'avg_retrieval_time': sum(baseline_retrieval_times) / len(baseline_retrieval_times),
                'avg_generation_time': sum(baseline_generation_times) / len(baseline_generation_times)
            }
            
            # Calculate overhead
            results['comparison'] = {
                'total_overhead': results['encrypted']['avg_total_time'] - results['baseline']['avg_total_time'],
                'total_overhead_pct': ((results['encrypted']['avg_total_time'] / results['baseline']['avg_total_time']) - 1) * 100,
                'retrieval_overhead': results['encrypted']['avg_retrieval_time'] - results['baseline']['avg_retrieval_time'],
                'retrieval_overhead_pct': ((results['encrypted']['avg_retrieval_time'] / results['baseline']['avg_retrieval_time']) - 1) * 100
            }
        
        return results
    
    def full_system_benchmark(self,
                             test_questions: List[str],
                             ground_truth_answers: List[str],
                             k_values: List[int] = [3, 5, 10]) -> Dict:
        """
        Run comprehensive system benchmark
        
        Args:
            test_questions: List of test questions
            ground_truth_answers: List of ground truth answers
            k_values: K values to test
            
        Returns:
            Dict with comprehensive benchmark results
        """
        logger.info("Running full system benchmark")
        
        results = {
            'num_questions': len(test_questions),
            'k_values_tested': k_values,
            'results_by_k': {}
        }
        
        for k in k_values:
            logger.info(f"Testing with K={k}")
            
            # Run evaluation
            eval_results = self.rag_system.evaluate(
                test_questions,
                ground_truth_answers,
                top_k=k
            )
            
            # Extract answers
            predicted_answers = [r['answer'] for r in eval_results['results']]
            
            # Calculate answer quality metrics
            answer_metrics = self.metrics.calculate_answer_metrics(
                predicted_answers,
                ground_truth_answers
            )
            
            results['results_by_k'][f'k_{k}'] = {
                'performance': {
                    'avg_total_time': eval_results['avg_time_per_question'],
                    'avg_retrieval_time': eval_results['avg_retrieval_time'],
                    'avg_generation_time': eval_results['avg_generation_time']
                },
                'quality': answer_metrics
            }
        
        return results
