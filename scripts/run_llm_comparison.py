"""Minimal LLM comparison for thesis.

Runs the same strict RAG prompt+pipeline with two Ollama models (llama2 vs mistral)
on a small set of questions (10-20) and writes JSONL results.

Scoring focuses on:
1) Whether the answer stays within context (no hallucination indicators).
2) Whether it avoids non-answers.

Note: Automatic judging is heuristic. For the paper, you can additionally sample
results for manual review.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Tuple
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.encryption import AESEncryption
from src.embedding import EmbeddingModel
from src.retrieval import VectorStore, Retriever
from src.llm import OllamaLLM
from src.rag_pipeline.rag_system import RAGSystem


BANNED_PATTERNS = (
    r"\bi don't know\b",
    r"\bi do not know\b",
    r"\bnot found\b",
    r"\bno information\b",
    r"\bno clear answer\b",
)


def _load_queries(path: str, limit: int) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        qs = [line.strip() for line in f if line.strip()]
    return qs[:limit]


def _heuristic_is_context_grounded(answer: str, context: str) -> bool:
    """Very small heuristic: check answer doesn't introduce many tokens not in context.

    This is not perfect, but it catches obvious hallucinations where the answer is
    long and mentions unrelated entities.
    """
    a = (answer or "").strip().lower()
    c = (context or "").strip().lower()
    if not a:
        return False
    if any(re.search(p, a) for p in BANNED_PATTERNS):
        return True  # non-answer is "safe" (not hallucination) but will be scored lower elsewhere

    # Token overlap ratio.
    a_tokens = [t for t in re.findall(r"[\w\u4e00-\u9fff]+", a) if len(t) >= 2]
    c_tokens = set(t for t in re.findall(r"[\w\u4e00-\u9fff]+", c) if len(t) >= 2)
    if not a_tokens:
        return False
    overlap = sum(1 for t in a_tokens if t in c_tokens)
    ratio = overlap / max(1, len(a_tokens))

    # If answer is short, be lenient.
    if len(a_tokens) <= 12:
        return ratio >= 0.25
    return ratio >= 0.35


def _score_one(answer: str, context: str) -> Tuple[int, str]:
    """Return (score, reason). Higher means better."""

    if not answer:
        return 0, "empty"

    a = answer.strip().lower()

    # Non-answer
    if any(re.search(p, a) for p in BANNED_PATTERNS):
        return 1, "fallback/non-answer"

    grounded = _heuristic_is_context_grounded(answer, context)
    if not grounded:
        return 0, "likely hallucination / low overlap with context"

    # Grounded answer
    return 2, "grounded in context"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run minimal llama2 vs mistral comparison")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--key_file", type=str, default="encryption.key")
    parser.add_argument(
        "--generate_key",
        action="store_true",
        help="If the key file does not exist, generate a new AES key and save it.",
    )
    parser.add_argument("--queries_file", type=str, required=True)
    parser.add_argument("--collection_name", type=str, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens per answer (keep small for CPU-only).")
    parser.add_argument("--timeout", type=int, default=300, help="Ollama request timeout seconds.")
    parser.add_argument("--top_k", type=int, default=None, help="Retriever top_k (default from config if not set)")
    parser.add_argument("--start_index", type=int, default=0, help="Start from this query index (0-based) for resume")
    parser.add_argument(
        "--per_question_timeout",
        type=int,
        default=180,
        help="Soft timeout per question (seconds). If exceeded, store I don't know for remaining model calls.",
    )
    parser.add_argument(
        "--force_copy_storage",
        action="store_true",
        help="Work around Windows local Qdrant locks by copying storage_path to a temp dir and reading from there.",
    )
    parser.add_argument(
        "--no_force_copy_storage",
        action="store_true",
        help="Disable the storage copy workaround.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    limit = int(args.limit)
    queries = _load_queries(args.queries_file, limit=limit)

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    output = args.output or os.path.join(out_dir, f"llm_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

    # Encryption key: required to decrypt ciphertext payloads from Qdrant.
    # If missing, we can optionally generate a new one (useful for fresh ingest).
    enc = AESEncryption(key_size=config["encryption"]["key_size"])
    if not os.path.exists(args.key_file):
        if args.generate_key:
            os.makedirs(os.path.dirname(args.key_file) or ".", exist_ok=True)
            enc.generate_key()
            enc.save_key(args.key_file)
            print(f"Generated new encryption key: {args.key_file}")
        else:
            raise FileNotFoundError(
                f"Encryption key not found: {args.key_file}. "
                f"Pass --key_file <path> to the existing key used during ingest, "
                f"or use --generate_key only if you will re-ingest documents with the new key."
            )
    else:
        enc.load_key(args.key_file)

    # NOTE: Embedding model load is expensive on CPU; load once.
    em = EmbeddingModel(model_name=config["embedding"]["model_name"])

    collection = args.collection_name or config["vector_db"]["collection_name"]

    storage_path = config["vector_db"]["storage_path"]

    # Workaround for Windows local Qdrant locks (.lock cannot be removed if another process holds it).
    # Strategy: copy the storage folder to a temp directory and open a local client on the copy.
    # This is safe for read-only workloads (comparison/evaluation) and doesn't mutate the main storage.
    use_copy = bool(args.force_copy_storage) and not bool(args.no_force_copy_storage)
    if use_copy and not os.path.isabs(storage_path):
        # keep relative paths relative to project root
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        storage_path_abs = os.path.abspath(os.path.join(proj_root, storage_path))
    else:
        storage_path_abs = storage_path

    if use_copy:
        import tempfile
        import shutil

        tmp_root = tempfile.mkdtemp(prefix="qdrant_storage_copy_")
        copied_storage = os.path.join(tmp_root, "qdrant_storage")
        try:
            if os.path.exists(storage_path_abs):
                shutil.copytree(storage_path_abs, copied_storage, dirs_exist_ok=True)
                storage_path = copied_storage
                print(f"[qdrant] Using copied local storage for lock-free read: {storage_path}")
        except Exception as e:
            print(f"[qdrant] Failed to copy storage for lock workaround ({e}); falling back to original storage.")

    vs = VectorStore(
        collection_name=collection,
        dimension=em.get_dimension(),
        distance_metric=config["vector_db"]["distance_metric"],
        storage_path=storage_path,
        host=config["vector_db"].get("host"),
        port=config["vector_db"].get("port"),
    )

    try:
        retriever = Retriever(em, vs, enc)

        llama2 = OllamaLLM(model_name="llama2", base_url=config["llm"]["base_url"], timeout=int(args.timeout))
        mistral = OllamaLLM(model_name="mistral", base_url=config["llm"]["base_url"], timeout=int(args.timeout))

        rag_llama2 = RAGSystem(
            retriever=retriever,
            llm_client=llama2,
            llm_name="llama2",
            prompt_template=config["rag"]["prompt_template"],
            max_context_length=config["rag"]["max_context_length"],
            reranker=None,
        )

        reranker = None
        if config.get("rerank", {}).get("enabled", False):
            from src.rag_pipeline.rerank import LocalReranker

            reranker = LocalReranker(
                max_candidates=config["rerank"].get("max_candidates", 20),
                min_score=config["rerank"].get("min_score", 0.0),
            )

        rag_mistral = RAGSystem(
            retriever=retriever,
            llm_client=mistral,
            llm_name="mistral",
            prompt_template=config["rag"]["prompt_template"],
            max_context_length=config["rag"]["max_context_length"],
            reranker=reranker,
        )

        stats = {"llama2_wins": 0, "mistral_wins": 0, "ties": 0, "total": 0}

        with open(output, "w", encoding="utf-8", buffering=1) as f:
            qlist = queries[int(args.start_index):]
            for idx, q in enumerate(qlist, start=int(args.start_index) + 1):
                t0 = time.time()
                print(f"[{idx}/{len(queries)}] question: {q[:120]}")

                top_k = int(args.top_k) if args.top_k is not None else int(config["retrieval"].get("default_top_k", 15))

                err = None
                try:
                    used_chunks = retriever.retrieve(q, top_k=top_k)
                    if used_chunks is None:
                        used_chunks = []
                except Exception as e:
                    used_chunks = []
                    err = f"retrieval_error: {e}"

                # Build context string with the same logic as RAGSystem (respect max_context_length)
                ctx_parts = []
                cur_len = 0
                for ch in used_chunks:
                    t = (ch.get("text") or "") if isinstance(ch, dict) else str(ch)
                    if not t:
                        continue
                    if cur_len + len(t) > int(config["rag"].get("max_context_length", 2000)):
                        remain = int(config["rag"].get("max_context_length", 2000)) - cur_len
                        if remain > 0:
                            ctx_parts.append(t[:remain])
                            cur_len += remain
                        break
                    ctx_parts.append(t)
                    cur_len += len(t)

                context = "\n\n".join(ctx_parts)

                # Prepare the strict prompt once
                prompt = config["rag"]["prompt_template"].format(context=context, question=q)

                l_ans = "I don't know"
                m_ans = "I don't know"

                try:
                    if time.time() - t0 <= int(args.per_question_timeout):
                        l_ans = llama2.generate(prompt, temperature=0.0, max_tokens=int(args.max_tokens)).text or "I don't know"
                except Exception as e:
                    err = (err + "; " if err else "") + f"llama2_error: {e}"

                try:
                    if time.time() - t0 <= int(args.per_question_timeout):
                        m_ans = mistral.generate(prompt, temperature=0.0, max_tokens=int(args.max_tokens)).text or "I don't know"
                except Exception as e:
                    err = (err + "; " if err else "") + f"mistral_error: {e}"

                # Diagnostics
                diag_common = {
                    "num_chunks_retrieved": len(used_chunks),
                    "retrieval_empty": len(used_chunks) == 0 or len(context.strip()) == 0,
                    "rerank_enabled": bool(config.get("rerank", {}).get("enabled", False)),
                    "context_length": len(context),
                }

                s_l, r_l = _score_one(l_ans, context)
                s_m, r_m = _score_one(m_ans, context)

                if s_m > s_l:
                    better = "mistral"
                    reason = f"mistral better: {r_m}; llama2: {r_l}"
                    stats["mistral_wins"] += 1
                elif s_l > s_m:
                    better = "llama2"
                    reason = f"llama2 better: {r_l}; mistral: {r_m}"
                    stats["llama2_wins"] += 1
                else:
                    better = "tie"
                    reason = f"tie: llama2={r_l}; mistral={r_m}"
                    stats["ties"] += 1

                stats["total"] += 1

                row: Dict = {
                    "question": q,
                    "llama2_answer": l_ans,
                    "mistral_answer": m_ans,
                    "better_model": better,
                    "reason": reason,
                    "error": err,
                    "rag_diagnostics": {
                        "llama2": diag_common,
                        "mistral": diag_common,
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

        win_rate = stats["mistral_wins"] / max(1, stats["total"])
        print("Wrote:", output)
        print("Summary:", stats)
        print("Mistral win rate:", round(win_rate * 100, 2), "%")
        return 0

    finally:
        # Explicitly close local qdrant client to release storage lock on Windows.
        try:
            if hasattr(vs, "client") and vs.client is not None:
                vs.client.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
