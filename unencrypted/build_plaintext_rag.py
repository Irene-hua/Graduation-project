"""Build a RAGSystem that points to the PLAINTEXT baseline Qdrant.

This mirrors `src.rag_pipeline.rag_system._build_runtime` but:
- uses an isolated local Qdrant storage path under `unencrypted/`
- uses a plaintext collection
- still constructs an `AESEncryption` instance (to keep Retriever signature identical),
  but plaintext payload means no decrypt cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import os
import yaml

from src.encryption import AESEncryption
from src.llm import OllamaLLM
from src.rag_pipeline.rag_system import RAGSystem
from src.rag_pipeline.rerank import LocalReranker
from src.retrieval import Retriever, VectorStore


def build_plaintext_runtime(
    *,
    config_path: str = 'config/config.yaml',
    key_file: str = 'encryption.key',
    collection_name: str = 'plaintext_documents_lihua_world',
    storage_path: str = 'unencrypted/qdrant_storage_plaintext',
    allow_empty_collection: bool = False,
) -> Tuple[RAGSystem, None]:
    project_root = Path(__file__).resolve().parent.parent
    cfg = Path(config_path)
    if not cfg.is_absolute():
        cfg = (project_root / cfg).resolve()

    with open(cfg, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Keep same encryption object to keep pipeline wiring identical.
    encryption = AESEncryption(key_size=config['encryption']['key_size'])
    kf = Path(key_file)
    if not kf.is_absolute():
        kf = (project_root / kf).resolve()
    if not os.path.exists(kf):
        raise FileNotFoundError(f'Encryption key not found: {kf}')
    encryption.load_key(str(kf))

    from src.embedding import EmbeddingModel

    embedding_model = EmbeddingModel(model_name=config['embedding']['model_name'])

    sp = Path(storage_path)
    if not sp.is_absolute():
        sp = (project_root / sp).resolve()

    vector_store = VectorStore(
        collection_name=collection_name,
        dimension=embedding_model.get_dimension(),
        distance_metric=config['vector_db']['distance_metric'],
        storage_path=str(sp),
        host=None,
        port=None,
    )

    info = vector_store.get_collection_info()
    if (info.get('points_count', 0) or 0) == 0 and not allow_empty_collection:
        raise RuntimeError(
            f"Plaintext collection is empty: collection='{collection_name}', storage_path='{sp}'. "
            "Please run unencrypted/ingest_plaintext.py first."
        )

    retriever = Retriever(embedding_model, vector_store, encryption)

    llm_name = config.get('llm', {}).get('default_model') or config['llm'].get('model_name', 'mistral')
    llm_client = OllamaLLM(model_name=llm_name, base_url=config['llm']['base_url'])
    if not llm_client.is_available():
        raise RuntimeError('Ollama server not available. Please start Ollama first: ollama serve')

    reranker = None
    if config.get('rerank', {}).get('enabled', False):
        reranker = LocalReranker(
            max_candidates=config['rerank'].get('max_candidates', 20),
            min_score=config['rerank'].get('min_score', 0.0),
        )

    rag_system = RAGSystem(
        retriever=retriever,
        llm_client=llm_client,
        llm_name=llm_name,
        prompt_template=config['rag']['prompt_template'],
        max_context_length=config['rag']['max_context_length'],
        reranker=reranker,
    )
    return rag_system, None
