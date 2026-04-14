# Unencrypted baseline (for performance comparison)

This directory contains an **isolated**, plaintext baseline pipeline used to compare against the project's **encrypted** RAG pipeline.

**Contract / non-goals**
- It must **not** modify or break the existing encrypted RAG system.
- It must **not** reuse the existing `qdrant_storage/`.
- All new artifacts (scripts, Qdrant storage, results, docs) live under `unencrypted/`.

## What is different?
Exactly one thing: **chunks are stored as plaintext in Qdrant payload** (`text`) instead of encrypted (`ciphertext` + `nonce`).

Everything else is reused to keep the comparison fair:
- same `DocumentParser` / `TextChunker`
- same embedding model and settings
- same retriever logic and result format
- same (optional) reranker and prompt
- same Ollama LLM caller

## Entry points
- `unencrypted/ingest_plaintext.py`: ingest `data/raw/LiHua-World` into an isolated plaintext Qdrant storage.
- `unencrypted/build_plaintext_rag.py`: builds a `RAGSystem` instance that points at the plaintext collection.
- `unencrypted/bench/run_perf_compare.py`: runs the benchmark comparing **encrypted vs plaintext** and produces a report.

## Outputs
Benchmark results are written under:

- `unencrypted/results/<run_id>/...`

and include raw samples (JSONL/CSV), summary stats (JSON/MD), and a paper-ready report.

