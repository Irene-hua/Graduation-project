#!/usr/bin/env python3
"""
Run RAG System
Interactive question-answering using the RAG pipeline
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import yaml
import logging

from src.encryption import AESEncryption
from src.embedding import EmbeddingModel
from src.retrieval import VectorStore, Retriever
from src.llm import OllamaClient
from src.rag_pipeline import RAGSystem
from src.audit import AuditLogger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run RAG system for Q&A')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--key_file', type=str, default='encryption.key',
                       help='Path to encryption key file')
    parser.add_argument('--question', type=str,
                       help='Single question to answer (if not provided, runs interactive mode)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of chunks to retrieve')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='LLM temperature')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Initializing RAG system")
    
    # Initialize audit logger
    audit_logger = None
    if config['audit']['enabled']:
        audit_logger = AuditLogger(
            log_directory=config['audit']['log_directory'],
            log_level=config['audit']['log_level'],
            enable_integrity_check=config['audit']['integrity_check']
        )
        audit_logger.log_system_access('user', 'initialize_rag_system')
    
    # Initialize encryption
    logger.info(f"Loading encryption key from {args.key_file}")
    encryption = AESEncryption(key_size=config['encryption']['key_size'])
    
    if not os.path.exists(args.key_file):
        logger.error(f"Encryption key not found: {args.key_file}")
        logger.error("Please run ingest_documents.py first to generate the key")
        return
    
    encryption.load_key(args.key_file)
    
    # Initialize embedding model
    logger.info("Loading embedding model")
    embedding_model = EmbeddingModel(
        model_name=config['embedding']['model_name']
    )
    
    # Initialize vector store
    logger.info("Connecting to vector database")
    vector_store = VectorStore(
        collection_name=config['vector_db']['collection_name'],
        dimension=embedding_model.get_dimension(),
        distance_metric=config['vector_db']['distance_metric'],
        storage_path=config['vector_db']['storage_path']
    )
    
    info = vector_store.get_collection_info()
    logger.info(f"Collection has {info['points_count']} documents")
    
    if info['points_count'] == 0:
        logger.error("Vector database is empty. Please run ingest_documents.py first")
        return
    
    # Initialize retriever
    retriever = Retriever(embedding_model, vector_store, encryption)
    
    # Initialize LLM
    logger.info("Connecting to Ollama")
    llm_client = OllamaClient(
        base_url=config['llm']['base_url'],
        model_name=config['llm']['model_name']
    )
    
    if not llm_client.is_available():
        logger.error("Ollama server not available. Please start Ollama first:")
        logger.error("  ollama serve")
        return
    
    # Initialize RAG system
    rag_system = RAGSystem(
        retriever=retriever,
        llm_client=llm_client,
        prompt_template=config['rag']['prompt_template'],
        max_context_length=config['rag']['max_context_length']
    )
    
    logger.info("RAG system ready!")
    
    # Process question(s)
    if args.question:
        # Single question mode
        process_question(rag_system, args.question, args.top_k, args.temperature, audit_logger)
    else:
        # Interactive mode
        print("\n" + "="*50)
        print("Interactive RAG Q&A System")
        print("Type 'quit' or 'exit' to stop")
        print("="*50 + "\n")
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                process_question(rag_system, question, args.top_k, args.temperature, audit_logger)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                if audit_logger:
                    audit_logger.log_error('question_processing', str(e))


def process_question(rag_system, question, top_k, temperature, audit_logger=None):
    """Process a single question"""
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    # Log query
    if audit_logger:
        audit_logger.log_query(question)
    
    # Answer question
    result = rag_system.answer_question(
        question=question,
        top_k=top_k,
        temperature=temperature
    )
    
    # Log model invocation
    if audit_logger:
        audit_logger.log_model_invocation(
            model_name=rag_system.llm_client.model_name,
            inference_time=result['generation_time']
        )
    
    # Display answer
    print(f"\nAnswer: {result['answer']}")
    print(f"\nRetrieved {result['num_chunks_retrieved']} chunks")
    print(f"Retrieval time: {result['retrieval_time']:.3f}s")
    print(f"Generation time: {result['generation_time']:.3f}s")
    print(f"Total time: {result['total_time']:.3f}s")
    
    # Show sources (prefer used_chunks which are actually passed to the LLM)
    used = result.get('used_chunks') or result.get('context_chunks') or []
    if used:
        print("\nSources:")
        try:
            nchunks = len(used)
        except Exception:
            nchunks = 0
        print(f"  (chunks used for generation: {nchunks})")
        for i, chunk in enumerate(used[:top_k], 1):
            source = chunk.get('metadata', {}).get('source_file', 'unknown')
            score = chunk.get('score', 0)
            chunk_id = chunk.get('metadata', {}).get('chunk_id', 'unknown')
            preview = (chunk.get('text') or '')[:120].replace('\n', ' ')
            print(f"  {i}. {source} (score: {score:.3f}) chunk_id={chunk_id} preview='{preview}...' ")


if __name__ == '__main__':
    main()
