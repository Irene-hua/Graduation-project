#!/usr/bin/env python3
"""
Benchmark Script
Run comprehensive benchmarks on RAG system
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import yaml
import logging
import json
from pathlib import Path

from src.encryption import AESEncryption
from src.embedding import EmbeddingModel
from src.retrieval import VectorStore, Retriever
from src.llm import OllamaClient
from src.rag_pipeline import RAGSystem
from src.evaluation import Benchmarking

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Benchmark RAG system')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--key_file', type=str, default='encryption.key',
                       help='Path to encryption key file')
    parser.add_argument('--test_queries', type=str, required=True,
                       help='Path to file with test queries (one per line)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--benchmark_type', type=str, 
                       choices=['k_values', 'embedding_models', 'full'],
                       default='k_values',
                       help='Type of benchmark to run')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test queries
    with open(args.test_queries, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(queries)} test queries")
    
    # Initialize components
    logger.info("Initializing RAG system")
    
    encryption = AESEncryption(key_size=config['encryption']['key_size'])
    encryption.load_key(args.key_file)
    
    embedding_model = EmbeddingModel(
        model_name=config['embedding']['model_name']
    )
    
    vector_store = VectorStore(
        collection_name=config['vector_db']['collection_name'],
        dimension=embedding_model.get_dimension(),
        distance_metric=config['vector_db']['distance_metric'],
        storage_path=config['vector_db']['storage_path']
    )
    
    retriever = Retriever(embedding_model, vector_store, encryption)
    
    llm_client = OllamaClient(
        base_url=config['llm']['base_url'],
        model_name=config['llm']['model_name']
    )
    
    rag_system = RAGSystem(
        retriever=retriever,
        llm_client=llm_client,
        prompt_template=config['rag']['prompt_template'],
        max_context_length=config['rag']['max_context_length']
    )
    
    # Initialize benchmarking
    benchmark = Benchmarking(rag_system)
    
    results = {}
    
    # Run benchmarks based on type
    if args.benchmark_type == 'k_values':
        logger.info("Benchmarking different K values")
        k_values = config['retrieval']['top_k_values']
        results = benchmark.benchmark_retrieval_k_values(queries, k_values)
        
    elif args.benchmark_type == 'embedding_models':
        logger.info("Benchmarking different embedding models")
        models = config['embedding']['alternative_models']
        test_texts = queries[:10]  # Use subset for model comparison
        results = benchmark.benchmark_embedding_models(models, test_texts)
        
    elif args.benchmark_type == 'full':
        logger.info("Running full system benchmark")
        # For full benchmark, we need ground truth answers
        # This is a simplified version
        k_values = config['retrieval']['top_k_values']
        
        for k in k_values:
            logger.info(f"Testing with K={k}")
            query_results = []
            
            for query in queries[:10]:  # Limit to 10 queries for demo
                result = rag_system.answer_question(query, top_k=k)
                query_results.append({
                    'query': query,
                    'answer': result['answer'],
                    'retrieval_time': result['retrieval_time'],
                    'generation_time': result['generation_time'],
                    'total_time': result['total_time']
                })
            
            results[f'k_{k}'] = {
                'results': query_results,
                'avg_total_time': sum(r['total_time'] for r in query_results) / len(query_results),
                'avg_retrieval_time': sum(r['retrieval_time'] for r in query_results) / len(query_results),
                'avg_generation_time': sum(r['generation_time'] for r in query_results) / len(query_results)
            }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("Benchmark Results Summary")
    print("="*50)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
