"""Document ingestion utilities moved into src so RAG core lives under src.

This module exposes a `main` function that performs parsing, chunking,
encryption, embedding and storage. It's a direct refactor of the previous
scripts/ingest_documents.py script so higher-level code can import and reuse it.
"""
from __future__ import annotations

import argparse
import yaml
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from .document_parser import DocumentParser
from .text_chunker import TextChunker
from src.encryption import AESEncryption
from src.embedding import EmbeddingModel
from src.retrieval import VectorStore

logger = logging.getLogger(__name__)


def ingest_documents(
    input_dir: str,
    config_path: str = 'config/config.yaml',
    key_file: str = 'encryption.key',
    generate_key: bool = False,
    collection_name: Optional[str] = None,
    reset_collection: bool = False,
    log_file: Optional[str] = None,
):
    """Run document ingestion pipeline.

    Args:
        input_dir: Path to directory containing documents.
        config_path: Path to YAML config file.
        key_file: Path to encryption key file.
        generate_key: Whether to generate and save a new key.
        collection_name: Optional override for target vector collection.
        reset_collection: If True, reset the target collection before ingest.
        log_file: Optional path to append a YAML/JSONL ingest record for auditing.

    Returns:
        A dict summary with counts and collection info.
    """
    project_root = Path(__file__).resolve().parents[2]
    input_dir = Path(input_dir)
    if not input_dir.is_absolute():
        input_dir = (project_root / input_dir).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    key_file = Path(key_file)
    if not key_file.is_absolute():
        key_file = (project_root / key_file).resolve()
    log_path = Path(log_file).resolve() if log_file else None

    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger.info("Starting document ingestion process")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Resolved input dir: {input_dir}")

    # Step 1: Parse documents
    logger.info(f"Parsing documents from {input_dir}")
    if not input_dir.exists() or not input_dir.is_dir():
        candidates = [p for p in [project_root / 'data' / 'lihua_world', project_root / 'data' / 'single_test1', project_root / 'data' / 'raw'] if p.exists()]
        logger.error(f"Invalid directory: {input_dir}")
        logger.error(f"Available data directories under project root: {candidates}")
        return {}

    parser = DocumentParser()
    documents = parser.parse_directory(str(input_dir), recursive=True)
    logger.info(f"Parsed {len(documents)} documents")

    if not documents:
        logger.error("No documents found to ingest")
        return {}

    # Step 2: Chunk documents
    logger.info("Chunking documents")
    chunker = TextChunker(
        chunk_size=config['document_processing']['chunk_size'],
        chunk_overlap=config['document_processing']['chunk_overlap']
    )
    chunks = chunker.chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    # Compute per-document hashes to improve provenance and enable dedup checks
    import hashlib
    doc_hashes = []
    for doc in documents:
        content = doc.get('content', '') or ''
        h = hashlib.sha256(content.encode('utf-8')).hexdigest()
        doc_hashes.append(h)

    # Step 3: Initialize encryption
    logger.info("Initializing encryption")
    encryption = AESEncryption(key_size=config['encryption']['key_size'])

    if generate_key or not key_file.exists():
        logger.info("Generating new encryption key")
        encryption.generate_key()
        encryption.save_key(str(key_file))
        logger.info(f"Encryption key saved to {key_file}")
    else:
        logger.info(f"Loading encryption key from {key_file}")
        encryption.load_key(str(key_file))

    # Step 4: Encrypt chunks
    logger.info("Encrypting chunks")
    encrypted_chunks = []
    for chunk in tqdm(chunks, desc="Encrypting"):
        doc_id = chunk.get('doc_id')
        source_path = None
        if isinstance(doc_id, int) and 0 <= doc_id < len(documents):
            source_path = documents[doc_id].get('filepath') or documents[doc_id].get('filename')
        content_hash = doc_hashes[doc_id] if (isinstance(doc_id, int) and 0 <= doc_id < len(doc_hashes)) else None

        ciphertext, nonce = encryption.encrypt(chunk['text'])
        metadata = {
            'source_file': chunk.get('source_file'),
            'source_path': source_path,
            'chunk_id': chunk.get('chunk_id'),
            'doc_id': doc_id,
            'content_hash': content_hash
        }
        for k, v in list(chunk.items()):
            if k.startswith('doc_') and k not in metadata:
                metadata[k] = v

        encrypted_chunks.append({
            'ciphertext': ciphertext,
            'nonce': nonce,
            'chunk_id': chunk['global_chunk_id'],
            'metadata': metadata
        })

    # Step 5: Generate embeddings
    logger.info("Generating embeddings")
    embedding_model = EmbeddingModel(
        model_name=config['embedding']['model_name']
    )
    texts = [chunk['text'] for chunk in chunks]
    embeddings = embedding_model.encode(
        texts,
        batch_size=config['embedding']['batch_size'],
        show_progress=True
    )
    logger.info(f"Generated embeddings with dimension {embeddings.shape[1]}")

    collection_name = collection_name or config['vector_db']['collection_name']
    import_started_at = datetime.now().isoformat(timespec='seconds')
    logger.info(f"Using collection: {collection_name}")
    logger.info(f"Import started at: {import_started_at}")

    # Step 6: Store in vector database
    logger.info("Storing in vector database")

    storage_path = config['vector_db']['storage_path']
    per_run_root = config.get('vector_db', {}).get('per_run_storage_root')
    if per_run_root:
        run_dir = datetime.now().strftime('%Y%m%d_%H%M%S')
        storage_path = str((project_root / per_run_root / run_dir).resolve())
        logger.info(f"per_run_storage_root enabled. Using per-run Qdrant storage: {storage_path}")

    vector_store = VectorStore(
        collection_name=collection_name,
        dimension=embedding_model.get_dimension(),
        distance_metric=config['vector_db']['distance_metric'],
        storage_path=storage_path,
        host=config['vector_db'].get('host'),
        port=config['vector_db'].get('port')
    )

    if reset_collection:
        logger.info(f"Resetting collection before ingest: {collection_name}")
        vector_store.reset_collection()

    metadata = [ec['metadata'] for ec in encrypted_chunks]
    point_ids = vector_store.add_vectors(embeddings, encrypted_chunks, metadata)
    logger.info(f"Added {len(point_ids)} vectors to database")
    collection_info = vector_store.get_collection_info()
    logger.info(f"Collection info after ingest: {collection_info}")

    ingest_record = {
        'timestamp': import_started_at,
        'input_dir': str(input_dir),
        'collection_name': collection_name,
        'reset_collection': bool(reset_collection),
        'documents_processed': len(documents),
        'chunks_created': len(chunks),
        'vectors_stored': len(point_ids),
        'collection_points_count': collection_info.get('points_count'),
        'key_file': str(key_file),
    }
    logger.info(f"Ingest record: {ingest_record}")
    if log_path:
        with open(log_path, 'a', encoding='utf-8') as logf:
            logf.write(yaml.safe_dump([ingest_record], allow_unicode=True, sort_keys=False))

    logger.info("="*50)
    logger.info("Ingestion complete!")
    logger.info(f"Documents processed: {len(documents)}")
    logger.info(f"Chunks created: {len(chunks)}")
    logger.info(f"Vectors stored: {len(point_ids)}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Encryption key: {key_file}")
    logger.info("="*50)

    return ingest_record


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
    parser.add_argument('--collection_name', type=str, default=None,
                       help='Override Qdrant collection name for this import')
    parser.add_argument('--reset_collection', action='store_true',
                       help='Delete and recreate the target collection before ingesting')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Optional path to append a JSONL ingest log record')

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ingest_documents(
        input_dir=args.input_dir,
        config_path=args.config,
        key_file=args.key_file,
        generate_key=args.generate_key,
        collection_name=args.collection_name,
        reset_collection=args.reset_collection,
        log_file=args.log_file,
    )


if __name__ == '__main__':
    main()

