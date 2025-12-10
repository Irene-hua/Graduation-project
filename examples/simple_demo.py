#!/usr/bin/env python3
"""
Simple Demo - Minimal example of using the RAG system
This demonstrates the core functionality in a simple script
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Step 1: Initialize encryption
print("Step 1: Setting up encryption...")
from src.encryption import AESEncryption

encryption = AESEncryption(key_size=256)
encryption.generate_key()
print("✓ Encryption initialized")

# Step 2: Prepare sample documents
print("\nStep 2: Preparing sample documents...")
sample_docs = [
    {
        'content': 'Python is a high-level programming language. It is widely used for web development, data science, and artificial intelligence.',
        'filename': 'python.txt'
    },
    {
        'content': 'Machine learning is a subset of AI that enables computers to learn from data without explicit programming.',
        'filename': 'ml.txt'
    },
    {
        'content': 'Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language.',
        'filename': 'nlp.txt'
    }
]
print(f"✓ Prepared {len(sample_docs)} documents")

# Step 3: Chunk documents
print("\nStep 3: Chunking documents...")
from src.document_processing import TextChunker

chunker = TextChunker(chunk_size=200, chunk_overlap=30)
chunks = chunker.chunk_documents(sample_docs)
print(f"✓ Created {len(chunks)} chunks")

# Step 4: Encrypt chunks
print("\nStep 4: Encrypting chunks...")
encrypted_chunks = []
for chunk in chunks:
    ciphertext, nonce = encryption.encrypt(chunk['text'])
    encrypted_chunks.append({
        'ciphertext': ciphertext,
        'nonce': nonce,
        'chunk_id': chunk['global_chunk_id'],
        'metadata': {
            'source_file': chunk['source_file'],
            'doc_id': chunk['doc_id']
        }
    })
print(f"✓ Encrypted {len(encrypted_chunks)} chunks")

# Step 5: Generate embeddings
print("\nStep 5: Generating embeddings...")
print("(This may take a moment on first run as it downloads the model)")
from src.embedding import EmbeddingModel

embedding_model = EmbeddingModel('sentence-transformers/all-MiniLM-L6-v2')
texts = [chunk['text'] for chunk in chunks]
embeddings = embedding_model.encode(texts, show_progress=False)
print(f"✓ Generated embeddings with dimension {embeddings.shape[1]}")

# Step 6: Store in vector database
print("\nStep 6: Storing in vector database...")
from src.retrieval import VectorStore

vector_store = VectorStore(
    collection_name='demo_collection',
    dimension=embedding_model.get_dimension(),
    storage_path='./demo_qdrant'
)

metadata = [ec['metadata'] for ec in encrypted_chunks]
point_ids = vector_store.add_vectors(embeddings, encrypted_chunks, metadata)
print(f"✓ Stored {len(point_ids)} vectors")

# Step 7: Test retrieval
print("\nStep 7: Testing retrieval...")
from src.retrieval import Retriever

retriever = Retriever(embedding_model, vector_store, encryption)

test_query = "What is machine learning?"
results = retriever.retrieve(test_query, top_k=2)

print(f"✓ Retrieved {len(results)} results for query: '{test_query}'")
print("\nResults:")
for i, result in enumerate(results, 1):
    print(f"\n{i}. Score: {result['score']:.3f}")
    print(f"   Text: {result['text'][:100]}...")
    print(f"   Source: {result['metadata']['source_file']}")

# Step 8: Cleanup
print("\n" + "="*60)
print("Demo complete! Cleaning up...")
vector_store.delete_collection()
print("✓ Cleaned up temporary data")

print("\n" + "="*60)
print("Summary:")
print(f"- Processed {len(sample_docs)} documents")
print(f"- Created {len(chunks)} chunks")
print(f"- Encrypted all data")
print(f"- Generated {embeddings.shape[1]}-dimensional embeddings")
print(f"- Successfully retrieved relevant content")
print("="*60)

print("\nNext steps:")
print("1. Try with your own documents using scripts/ingest_documents.py")
print("2. Set up Ollama for full RAG Q&A with scripts/run_rag.py")
print("3. Run benchmarks to evaluate performance")
