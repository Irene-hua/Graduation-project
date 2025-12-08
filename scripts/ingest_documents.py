#!/usr/bin/env python3
"""
Document Ingestion Script
Parse, chunk, encrypt, and store documents in vector database
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import yaml
import logging
from pathlib import Path
from tqdm import tqdm

from src.document_processing import DocumentParser, TextChunker
from src.encryption import AESEncryption
from src.embedding import EmbeddingModel
from src.retrieval import VectorStore

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Ingest documents into RAG system')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing documents to ingest')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--key_file', type=str, default='encryption.key',
                       help='Path to encryption key file')
    parser.add_argument('--generate_key', action='store_true',
                       help='Generate new encryption key')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Starting document ingestion process")
    
    # Step 1: Parse documents
    logger.info(f"Parsing documents from {args.input_dir}")
    parser = DocumentParser()
    documents = parser.parse_directory(args.input_dir, recursive=True)
    logger.info(f"Parsed {len(documents)} documents")
    
    if not documents:
        logger.error("No documents found to ingest")
        return
    
    # Step 2: Chunk documents
    logger.info("Chunking documents")
    chunker = TextChunker(
        chunk_size=config['document_processing']['chunk_size'],
        chunk_overlap=config['document_processing']['chunk_overlap']
    )
    chunks = chunker.chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Step 3: Initialize encryption
    logger.info("Initializing encryption")
    encryption = AESEncryption(key_size=config['encryption']['key_size'])
    
    if args.generate_key or not os.path.exists(args.key_file):
        logger.info("Generating new encryption key")
        encryption.generate_key()
        encryption.save_key(args.key_file)
        logger.info(f"Encryption key saved to {args.key_file}")
    else:
        logger.info(f"Loading encryption key from {args.key_file}")
        encryption.load_key(args.key_file)
    
    # Step 4: Encrypt chunks
    logger.info("Encrypting chunks")
    encrypted_chunks = []
    for chunk in tqdm(chunks, desc="Encrypting"):
        ciphertext, nonce = encryption.encrypt(chunk['text'])
        encrypted_chunks.append({
            'ciphertext': ciphertext,
            'nonce': nonce,
            'chunk_id': chunk['global_chunk_id'],
            'metadata': {
                'source_file': chunk['source_file'],
                'chunk_id': chunk['chunk_id'],
                'doc_id': chunk['doc_id']
            }
        })
    
    # Step 5: Generate embeddings
    logger.info("Generating embeddings")
    embedding_model = EmbeddingModel(
        model_name=config['embedding']['model_name']
    )
    
    # Get original texts for embedding
    texts = [chunk['text'] for chunk in chunks]
    embeddings = embedding_model.encode(
        texts,
        batch_size=config['embedding']['batch_size'],
        show_progress=True
    )
    logger.info(f"Generated embeddings with dimension {embeddings.shape[1]}")
    
    # Step 6: Store in vector database
    logger.info("Storing in vector database")
    vector_store = VectorStore(
        collection_name=config['vector_db']['collection_name'],
        dimension=embedding_model.get_dimension(),
        distance_metric=config['vector_db']['distance_metric'],
        storage_path=config['vector_db']['storage_path']
    )
    
    # Prepare metadata
    metadata = [ec['metadata'] for ec in encrypted_chunks]
    
    # Add to database
    point_ids = vector_store.add_vectors(embeddings, encrypted_chunks, metadata)
    logger.info(f"Added {len(point_ids)} vectors to database")
    
    # Print summary
    logger.info("="*50)
    logger.info("Ingestion complete!")
    logger.info(f"Documents processed: {len(documents)}")
    logger.info(f"Chunks created: {len(chunks)}")
    logger.info(f"Vectors stored: {len(point_ids)}")
    logger.info(f"Collection: {config['vector_db']['collection_name']}")
    logger.info(f"Encryption key: {args.key_file}")
    logger.info("="*50)


if __name__ == '__main__':
    main()
