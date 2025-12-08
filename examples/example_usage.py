#!/usr/bin/env python3
"""
Example Usage of Privacy-Preserving RAG System
Demonstrates basic usage of all components
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.document_processing import DocumentParser, TextChunker
from src.encryption import AESEncryption
from src.embedding import EmbeddingModel
from src.retrieval import VectorStore, Retriever
from src.llm import OllamaClient
from src.rag_pipeline import RAGSystem
from src.audit import AuditLogger


def example_document_processing():
    """Example: Parse and chunk documents"""
    print("\n=== Example 1: Document Processing ===")
    
    # Create sample document
    sample_text = """
    Machine learning is a subset of artificial intelligence that focuses on 
    developing algorithms that can learn from data. Deep learning is a type 
    of machine learning that uses neural networks with multiple layers.
    """
    
    # Chunk the text
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_text(sample_text)
    
    print(f"Created {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"  - Chunk {chunk['chunk_id']}: {chunk['text'][:50]}...")


def example_encryption():
    """Example: Encrypt and decrypt text"""
    print("\n=== Example 2: Encryption ===")
    
    # Initialize encryption
    encryption = AESEncryption(key_size=256)
    encryption.generate_key()
    
    # Encrypt text
    plaintext = "This is sensitive information that needs to be protected."
    ciphertext, nonce = encryption.encrypt(plaintext)
    
    print(f"Original: {plaintext}")
    print(f"Encrypted: {ciphertext[:50]}...")
    
    # Decrypt
    decrypted = encryption.decrypt(ciphertext, nonce)
    print(f"Decrypted: {decrypted}")
    print(f"Match: {plaintext == decrypted}")


def example_embedding():
    """Example: Generate text embeddings"""
    print("\n=== Example 3: Embeddings ===")
    
    # Initialize model
    model = EmbeddingModel('sentence-transformers/all-MiniLM-L6-v2')
    
    # Encode texts
    texts = [
        "Machine learning is fascinating",
        "Deep learning uses neural networks",
        "I like pizza and pasta"
    ]
    
    embeddings = model.encode(texts)
    
    print(f"Encoded {len(texts)} texts")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Calculate similarity
    sim_1_2 = model.cosine_similarity(embeddings[0], embeddings[1])
    sim_1_3 = model.cosine_similarity(embeddings[0], embeddings[2])
    
    print(f"Similarity between text 1 and 2: {sim_1_2:.3f}")
    print(f"Similarity between text 1 and 3: {sim_1_3:.3f}")


def example_vector_store():
    """Example: Store and retrieve vectors"""
    print("\n=== Example 4: Vector Store ===")
    
    # Initialize
    import numpy as np
    
    vector_store = VectorStore(
        collection_name='example_collection',
        dimension=384,
        storage_path='./example_qdrant'
    )
    
    # Create sample vectors
    vectors = np.random.rand(5, 384).astype(np.float32)
    
    # Create sample encrypted chunks
    encrypted_chunks = [
        {'ciphertext': f'cipher_{i}', 'nonce': f'nonce_{i}', 'chunk_id': i}
        for i in range(5)
    ]
    
    # Add to database
    ids = vector_store.add_vectors(vectors, encrypted_chunks)
    print(f"Added {len(ids)} vectors")
    
    # Search
    query_vector = np.random.rand(384).astype(np.float32)
    results = vector_store.search(query_vector, top_k=3)
    
    print(f"Retrieved {len(results)} results")
    for i, result in enumerate(results):
        print(f"  Result {i+1}: score={result['score']:.3f}")
    
    # Cleanup
    vector_store.delete_collection()


def example_audit_logging():
    """Example: Audit logging"""
    print("\n=== Example 5: Audit Logging ===")
    
    # Initialize audit logger
    logger = AuditLogger(log_directory='./example_logs')
    
    # Log various events
    logger.log_system_access('user123', 'login')
    logger.log_query('What is machine learning?', user='user123', results_count=5)
    logger.log_model_invocation('llama2', inference_time=2.5, tokens_generated=150)
    
    print("Logged 3 events")
    
    # Get statistics
    stats = logger.get_statistics()
    print(f"Total events: {stats['total_events']}")
    print(f"Event types: {stats['event_types']}")


def main():
    """Run all examples"""
    print("="*60)
    print("Privacy-Preserving RAG System - Example Usage")
    print("="*60)
    
    example_document_processing()
    example_encryption()
    
    print("\n" + "="*60)
    print("Note: Some examples require additional setup:")
    print("  - example_embedding(): downloads sentence transformer model")
    print("  - example_vector_store(): requires write permissions")
    print("  - example_audit_logging(): requires write permissions")
    print("="*60)
    
    # Run if user confirms
    response = input("\nRun all examples? (y/n): ").strip().lower()
    if response == 'y':
        example_embedding()
        example_vector_store()
        example_audit_logging()
        
        print("\n" + "="*60)
        print("All examples completed!")
        print("="*60)


if __name__ == '__main__':
    main()
