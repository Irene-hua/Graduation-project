#!/usr/bin/env python3
"""Plaintext ingestion script (baseline).

Mirror of `scripts/ingest_documents.py` but stores plaintext chunks in an isolated
Qdrant collection + storage path.

Important: this script does NOT touch the encrypted pipeline's Qdrant storage.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import argparse
import logging
import yaml
from tqdm import tqdm

# Ensure project root import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.document_processing import DocumentParser, TextChunker
from src.embedding import EmbeddingModel
from src.retrieval import VectorStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Ingest documents into PLAINTEXT baseline Qdrant')
    parser.add_argument('--input_dir', type=str, default='data/raw/LiHua-World', help='Directory containing raw docs')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config')
    parser.add_argument('--collection_name', type=str, default='plaintext_documents_lihua_world', help='Plaintext collection name')
    parser.add_argument('--storage_path', type=str, default='unencrypted/qdrant_storage_plaintext', help='Isolated local qdrant storage path')
    parser.add_argument('--reset_collection', action='store_true', help='Delete/recreate collection before ingesting')
    parser.add_argument('--log_jsonl', type=str, default='unencrypted/results/ingest_log.jsonl', help='Append ingest record as JSONL')

    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = (project_root / input_dir).resolve()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (project_root / cfg_path).resolve()

    # Load config
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Invalid input_dir: {input_dir}")

    logger.info('PLAINTEXT ingest starting')
    logger.info('Resolved input_dir=%s', input_dir)
    logger.info('Resolved config=%s', cfg_path)

    # Parse
    dp = DocumentParser()
    documents = dp.parse_directory(str(input_dir), recursive=True)
    logger.info('Parsed %d documents', len(documents))
    if not documents:
        raise RuntimeError('No documents found to ingest')

    # Chunk
    chunker = TextChunker(
        chunk_size=config['document_processing']['chunk_size'],
        chunk_overlap=config['document_processing']['chunk_overlap'],
    )
    chunks = chunker.chunk_documents(documents)
    logger.info('Created %d chunks', len(chunks))

    # Embeddings (same as encrypted pipeline)
    emb = EmbeddingModel(model_name=config['embedding']['model_name'])
    texts = [c['text'] for c in chunks]
    vectors = emb.encode(texts, batch_size=config['embedding']['batch_size'], show_progress=True)

    # Isolated storage
    storage_path = Path(args.storage_path)
    if not storage_path.is_absolute():
        storage_path = (project_root / storage_path).resolve()
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    vs = VectorStore(
        collection_name=args.collection_name,
        dimension=emb.get_dimension(),
        distance_metric=config['vector_db']['distance_metric'],
        storage_path=str(storage_path),
        host=None,
        port=None,
    )

    if args.reset_collection:
        vs.reset_collection()

    # Prepare payloads
    plaintext_chunks = []
    metadata = []

    import hashlib
    for i, ch in enumerate(tqdm(chunks, desc='Preparing payloads')):
        doc_id = ch.get('doc_id')
        source_path = None
        if isinstance(doc_id, int) and 0 <= doc_id < len(documents):
            source_path = documents[doc_id].get('filepath') or documents[doc_id].get('filename')

        content = (ch.get('text') or '')
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest() if content else None

        md = {
            'source_file': ch.get('source_file'),
            'source_path': source_path,
            'chunk_id': ch.get('chunk_id'),
            'doc_id': doc_id,
            'content_hash': content_hash,
        }
        for k, v in list(ch.items()):
            if k.startswith('doc_') and k not in md:
                md[k] = v

        plaintext_chunks.append({'text': ch['text'], 'chunk_id': ch.get('global_chunk_id', i)})
        metadata.append(md)

    # Upsert using plain payload by reusing VectorStore internals via a small adapter.
    # We call a dedicated method for plaintext to avoid changing the existing encrypted VectorStore API.
    point_ids = []
    batch_size = 100
    import uuid
    from qdrant_client.models import PointStruct

    points = []
    for v, pc, md in zip(vectors, plaintext_chunks, metadata):
        pid = str(uuid.uuid4())
        point_ids.append(pid)
        payload = {'text': pc['text'], 'chunk_id': pc.get('chunk_id')}
        payload.update(md)
        points.append(PointStruct(id=pid, vector=v.tolist(), payload=payload))

    for i in range(0, len(points), batch_size):
        vs.client.upsert(collection_name=vs.collection_name, points=points[i:i + batch_size])

    info = vs.get_collection_info()
    logger.info('PLAINTEXT ingest done. points=%d collection_info=%s', len(point_ids), info)

    # Log record
    log_path = Path(args.log_jsonl)
    if not log_path.is_absolute():
        log_path = (project_root / log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    rec = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'input_dir': str(input_dir),
        'collection_name': args.collection_name,
        'storage_path': str(storage_path),
        'reset_collection': bool(args.reset_collection),
        'documents_processed': len(documents),
        'chunks_created': len(chunks),
        'vectors_stored': len(point_ids),
        'collection_points_count': info.get('points_count'),
    }

    import json
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()

