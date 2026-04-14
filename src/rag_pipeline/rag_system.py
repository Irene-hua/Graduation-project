"""
RAG System
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import argparse
import logging
import os
import re
import time

import yaml

from src.audit import AuditLogger
from src.encryption import AESEncryption
from src.llm import BaseLLM, LLMResult, OllamaLLM
from src.rag_pipeline.rerank import LocalReranker
from src.retrieval import Retriever, VectorStore

logger = logging.getLogger(__name__)


class ContextBuilder:
    def __init__(
        self,
        max_chars: int = 4000,
        max_tokens: int = 1200,
        min_rerank_score: float = 0.2,
        keep_top_n: int = 2,
    ):
        self.max_chars = int(max_chars)
        self.max_tokens = int(max_tokens)
        self.min_rerank_score = float(min_rerank_score)
        self.keep_top_n = int(keep_top_n)

    @staticmethod
    def _token_estimate(text: str) -> int:
        t = (text or "").strip()
        if not t:
            return 0
        return max(1, int(len(t) / 4) + int(len(t.split()) / 2))

    def build(self, question: str, chunks: List[Dict], top_k: int):
        if not chunks:
            return "", []

        ranked = list(chunks)
        try:
            ranked.sort(key=lambda c: float(c.get("rerank_score", c.get("score", 0.0)) or 0.0), reverse=True)
        except Exception:
            pass

        parts: List[str] = []
        used: List[Dict] = []
        total_chars = 0
        total_tokens = 0
        seen_texts = set()

        limit = max(1, top_k)
        for idx, ch in enumerate(ranked[:limit], 1):
            score = float(ch.get("rerank_score", ch.get("score", 0.0)) or 0.0)
            # Drop low-quality chunks, but always keep the first few top-ranked chunks.
            if score < self.min_rerank_score and idx > self.keep_top_n:
                continue

            text = (ch.get("text") or "").strip()
            if not text:
                continue

            if text in seen_texts:
                continue
            seen_texts.add(text)

            md = ch.get("metadata") if isinstance(ch.get("metadata"), dict) else {}
            src = md.get("source_file") or md.get("source") or md.get("source_file_name") or "unknown_source"
            cid = md.get("chunk_id")
            provenance = f"[source: {src}" + (f" chunk_id: {cid}" if cid is not None else "") + "]"
            block = f"[{idx}] {provenance}\n{text}".strip()

            block_tokens = self._token_estimate(block)
            # Prevent truncating away critical evidence: always include the first keep_top_n chunks.
            if parts and (total_chars + len(block) > self.max_chars or total_tokens + block_tokens > self.max_tokens):
                if idx <= self.keep_top_n:
                    parts.append(block)
                    used.append({**ch, "text": text})
                break

            parts.append(block)
            used.append({**ch, "text": text})
            total_chars += len(block)
            total_tokens += block_tokens

        return "\n\n".join(parts), used


class LLMCaller:
    def __init__(self, client):
        self.client = client

    def generate(self, prompt: str, *, temperature: float = 0.7, max_tokens: Optional[int] = None):
        try:
            if isinstance(self.client, BaseLLM):
                res: LLMResult = self.client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
                return res.text
            return self.client.generate(prompt=prompt, temperature=temperature, max_tokens=max_tokens)["response"]
        except Exception as e:
            logger.error("LLM generation error: %s", e)
            return "[ERROR]"


@dataclass
class RAGContext:
    """Compact context container for standard result construction."""

    used_chunks: List[Dict]
    context_chunks: List[Dict]
    context_length: int
    retrieve_k: int
    retrieval_time: float


class RAGSystem:
    LLM_CONFIDENCE_FLOOR = 0.0

    def __init__(
        self,
        retriever,
        llm_client=None,
        *,
        llm_name: str = "mistral",
        prompt_template: Optional[str] = None,
        max_context_length: int = 2000,
        reranker=None,
        llm_base_url: str = "http://localhost:11434",
    ):
        self.retriever = retriever
        self.context_builder = ContextBuilder()
        if llm_client is None:
            llm_client = OllamaLLM(model_name=llm_name, base_url=llm_base_url)

        self.llm = LLMCaller(llm_client)
        self.reranker = reranker
        self.prompt_template = prompt_template or self._default_prompt()
        self.max_context_length = max_context_length
        self.llm_name = llm_name

    def _default_prompt(self):
        return (
            "You are a RAG question-answering assistant.\n"
            "Use the provided context as the primary source. You may reason if needed.\n"
            "If the context is incomplete, you may use reasonable inference and common sense to connect facts, but do NOT invent unsupported details.\n\n"
            "When the question asks WHO (people/entities), you MUST list the specific names/entities found in the context.\n"
            "Do NOT answer with vague phrases like 'all characters', 'everyone', or 'people in the context'.\n\n"
            "You MUST perform implicit reasoning silently before answering.\n"
            "If the answer involves a date, number, or entity, give the exact value.\n"
            "For time questions, you may compute dates/times when the context provides a reference timestamp (e.g., infer a next-day date).\n\n"
            "If the answer is not explicitly stated:\n"
            "- You MUST infer the answer using the context\n"
            "- Use only information that can be logically derived from the context\n"
            "- Do NOT introduce external facts\n\n"
            "You MUST attempt inference BEFORE saying \"I don't know\".\n\n"
            "Only reply with 'I don't know' if absolutely no relevant information exists in the context.\n"
            "If some relevant information exists, provide the best possible answer.\n\n"
            "Output format (STRICT):\n"
            "Answer: <final answer>\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n"
        )

    def build_prompt(self, question: str, context: str) -> str:
        """Build the final prompt."""
        return self.prompt_template.format(context=context, question=question)

    @staticmethod
    def is_valid_answer(answer: str) -> bool:
        if not answer:
            return False

        a = (answer or "").strip()
        if not a:
            return False

        al = re.sub(r"\s+", " ", a.lower()).strip()
        # Reject only fully-refusing minimal outputs.
        if al in {"i don't know", "answer: i don't know", "unknown"}:
            return False

        # Otherwise accept (including partial answers that mention uncertainty).
        return True

    @staticmethod
    def _compute_confidence(*, answer_text: str, rerank_top_scores: List[float], weak_answer: bool) -> float:
        """Compute a lightweight dynamic confidence for downstream evaluation."""
        top = float(rerank_top_scores[0]) if rerank_top_scores else 0.0
        conf = (
            0.6 * top
            + 0.2 * (1.0 if len((answer_text or "").strip()) > 20 else 0.0)
            + 0.2 * (1.0 if not weak_answer else 0.0)
        )
        return float(max(0.0, min(1.0, conf)))

    @staticmethod
    def _normalize_to_answer_format(text: str) -> str:
        """Format-only normalization: keep semantics, enforce single-line 'Answer:' output.

        - If the model returns 'Answer:' plus extra lines (e.g., 'Explanation:'), we keep ONLY the first Answer line.
        - If the model doesn't prefix with 'Answer:', we wrap the full text into a single Answer line.
        """
        t = (text or "").strip()
        if not t:
            return "Answer: I don't know"

        if t.lower().startswith("answer:"):
            lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
            return lines[0] if lines else "Answer: I don't know"

        # Keep the explanation inline after Answer if the model output already contains it.
        one_line = re.sub(r"\s+", " ", t).strip()
        return f"Answer: {one_line}"

    @staticmethod
    def _looks_like_who_question(question: str) -> bool:
        q = (question or "").strip().lower()
        return bool(re.search(r"\bwho\b", q))

    @staticmethod
    def _has_vague_who_answer(answer: str) -> bool:
        """Heuristic: detect non-specific answers for WHO questions.

        This doesn't inject any hardcoded names/logic. It only decides whether to re-ask the LLM
        to be more specific using the same context.
        """
        a = (answer or "").lower()
        if not a.startswith("answer:"):
            return False
        vague_markers = [
            "all the characters",
            "all characters",
            "everyone",
            "everybody",
            "people in the context",
            "all of them",
            "all the people",
        ]
        return any(m in a for m in vague_markers)

    def answer_question(self, question: str, top_k: int = 5, temperature: float = 0.2, max_tokens: Optional[int] = None):
        start = time.time()
        rerank_enabled = self.reranker is not None
        rerank_before_top1 = None
        rerank_after_top1 = None
        retrieve_k = max(top_k * 6, 30)
        rerank_top_scores: List[float] = []

        def _result(
            *,
            answer_text: str,
            path: str,
            reasoning_path: str,
            confidence: float,
            ctx: RAGContext,
            generation_time: float = 0.0,
            weak_answer: bool = False,
            retrieval_empty: bool = False,
            error: Optional[str] = None,
        ) -> Dict:
            """Build the standard result dict for this pipeline.

            Centralized to avoid drift across early-returns.
            """
            now_t = time.time()
            out = {
                "answer": answer_text,
                "path": path,
                "reasoning_path": reasoning_path,
                "time": now_t - start,
                "total_time": now_t - start,
                "retrieval_time": ctx.retrieval_time,
                "generation_time": generation_time,
                "num_chunks_retrieved": len(ctx.context_chunks or []),
                "used_chunks": ctx.used_chunks,
                "context_chunks": ctx.context_chunks,
                "confidence": confidence,
                "rerank_enabled": rerank_enabled,
                "rerank_before_top1": rerank_before_top1,
                "rerank_after_top1": rerank_after_top1,
                "rerank_top_scores": rerank_top_scores,
                "context_length": ctx.context_length,
                "retrieve_k": ctx.retrieve_k,
                "weak_answer": weak_answer,
                "retrieval_empty": retrieval_empty,
            }
            if error is not None:
                out["error"] = error
            return out

        try:
            chunks = self.retriever.retrieve(question, top_k=retrieve_k)
        except Exception as e:
            ctx = RAGContext(
                used_chunks=[],
                context_chunks=[],
                context_length=0,
                retrieve_k=retrieve_k,
                retrieval_time=0.0,
            )
            return _result(
                answer_text="[ERROR]",
                path="FAIL",
                reasoning_path="error",
                confidence=0.0,
                ctx=ctx,
                weak_answer=True,
                retrieval_empty=True,
                error=str(e),
            )

        retrieval_time = time.time() - start

        if self.reranker is not None:
            try:
                rerank_before_top1 = chunks[0].get("text") if chunks else None
                chunks = self.reranker.rerank(question, chunks, top_k=retrieve_k)
                rerank_after_top1 = chunks[0].get("text") if chunks else None
            except Exception as e:
                logger.debug("Reranker failed, fallback to retrieval order: %s", e)

        rerank_top_scores = [float(ch.get("rerank_score", ch.get("score", 0.0)) or 0.0) for ch in (chunks or [])[:3]]
        retrieval_quality_low = (not rerank_top_scores) or max(rerank_top_scores) < 0.1

        context, used_chunks = self.context_builder.build(question, chunks, top_k)
        context_length = len(context)
        ctx = RAGContext(
            used_chunks=used_chunks,
            context_chunks=chunks,
            context_length=context_length,
            retrieve_k=retrieve_k,
            retrieval_time=retrieval_time,
        )

        if context_length == 0:
            answer = "Answer: I don't know"
            return _result(
                answer_text=answer,
                path="FAIL",
                reasoning_path="no_context",
                confidence=0.0,
                ctx=ctx,
                weak_answer=True,
                retrieval_empty=True,
            )

        prompt = self.build_prompt(question, context)

        gen_start = time.time()
        answer = self.llm.generate(prompt, temperature=temperature, max_tokens=max_tokens)

        # Format normalization: avoid retries that reduce stability.
        answer = self._normalize_to_answer_format(answer)

        # If this is a WHO question and the model responded vaguely, retry once with a specificity reminder.
        if self._looks_like_who_question(question) and self._has_vague_who_answer(answer):
            specific_prompt = (
                "IMPORTANT: The question asks WHO. List the exact names/entities from the context. "
                "If multiple, return a comma-separated list. Do not use vague phrases.\n\n"
                + prompt
            )
            answer_retry = self.llm.generate(specific_prompt, temperature=temperature, max_tokens=max_tokens)
            answer_retry = self._normalize_to_answer_format(answer_retry)
            if self.is_valid_answer(answer_retry) and not self._has_vague_who_answer(answer_retry):
                answer = answer_retry

        generation_time = time.time() - gen_start
        retrieval_empty = (len(chunks) == 0) or (context_length == 0)

        if self.is_valid_answer(answer):
            # Keep confidence, but reduce its influence to avoid overfitting to heuristics.
            conf = 0.5 * self._compute_confidence(answer_text=answer, rerank_top_scores=rerank_top_scores, weak_answer=False)
            return _result(
                answer_text=answer,
                path="RAG",
                reasoning_path="llm",
                confidence=conf,
                ctx=ctx,
                generation_time=generation_time,
                weak_answer=False,
                retrieval_empty=retrieval_empty,
            )

        out = _result(
            answer_text="Answer: I don't know",
            path="FAIL",
            reasoning_path="llm",
            confidence=0.0,
            ctx=ctx,
            generation_time=generation_time,
            weak_answer=True,
            retrieval_empty=retrieval_empty,
        )

        if retrieval_quality_low:
            out["retrieval_quality_low"] = True
        return out


def _build_parser():
    parser = argparse.ArgumentParser(description='Run RAG system for Q&A')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--key_file', type=str, default='encryption.key', help='Path to encryption key file')
    parser.add_argument('--question', type=str, help='Single question to answer (if not provided, runs interactive mode)')
    parser.add_argument('--top_k', type=int, default=5, help='Number of chunks to retrieve')
    parser.add_argument('--temperature', type=float, default=0.2, help='LLM temperature')
    parser.add_argument('--collection_name', type=str, default=None, help='Override Qdrant collection name for this run')
    parser.add_argument('--allow_empty_collection', action='store_true', help='Allow running even if target Qdrant collection has 0 points.')
    return parser


def _build_runtime(
    config_path: str,
    key_file: str,
    collection_name: Optional[str] = None,
    allow_empty_collection: bool = False,
):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    audit_logger = None
    if config.get('audit', {}).get('enabled', False):
        audit_logger = AuditLogger(
            log_directory=config['audit']['log_directory'],
            log_level=config['audit']['log_level'],
            enable_integrity_check=config['audit']['integrity_check']
        )
        audit_logger.log_system_access('user', 'initialize_rag_system')

    encryption = AESEncryption(key_size=config['encryption']['key_size'])
    if not os.path.exists(key_file):
        raise FileNotFoundError(f'Encryption key not found: {key_file}')
    encryption.load_key(key_file)

    # Lazy import keeps rag_system importable even in minimal environments and avoids unused top-level imports.
    from src.embedding import EmbeddingModel
    embedding_model = EmbeddingModel(model_name=config['embedding']['model_name'])

    effective_collection = collection_name or config['vector_db']['collection_name']
    vector_store = VectorStore(
        collection_name=effective_collection,
        dimension=embedding_model.get_dimension(),
        distance_metric=config['vector_db']['distance_metric'],
        storage_path=config['vector_db']['storage_path'],
        host=config['vector_db'].get('host'),
        port=config['vector_db'].get('port')
    )
    info = vector_store.get_collection_info()
    if (info.get('points_count', 0) or 0) == 0 and not allow_empty_collection:
        try:
            existing = [c.name for c in vector_store.client.get_collections().collections]
        except Exception:
            existing = []

        raise RuntimeError(
            "Vector database collection is empty. "
            f"Target collection='{effective_collection}' has 0 points.\n\n"
            "Fix options:\n"
            f"  python -m scripts.ingest_documents --input_dir data\\single_test1 --collection_name {effective_collection}\n"
            "  or run without --collection_name to use default config collection.\n"
            "  or add --allow_empty_collection (debug only).\n\n"
            f"Existing collections: {existing or 'unknown/unavailable'}"
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
            min_score=config['rerank'].get('min_score', 0.0)
        )

    rag_system = RAGSystem(
        retriever=retriever,
        llm_client=llm_client,
        llm_name=llm_name,
        prompt_template=config['rag']['prompt_template'],
        max_context_length=config['rag']['max_context_length'],
        reranker=reranker
    )
    return rag_system, audit_logger


def process_question(rag_system, question, top_k, temperature, audit_logger=None):
    print(f"\nQuestion: {question}")
    print("-" * 50)

    if audit_logger:
        audit_logger.log_query(question)

    result = rag_system.answer_question(question=question, top_k=top_k, temperature=temperature)

    if audit_logger:
        model_obj = getattr(getattr(rag_system, 'llm', None), 'client', None)
        model_name = getattr(model_obj, 'model_name', 'unknown-model')
        audit_logger.log_model_invocation(model_name=model_name, inference_time=result['generation_time'])

    # result['answer'] is already a formatted string (typically starting with 'Answer:')
    print(f"\n{result['answer']}")
    print(f"\nRetrieved {result['num_chunks_retrieved']} chunks")
    print(f"Retrieval time: {result['retrieval_time']:.3f}s")
    print(f"Generation time: {result['generation_time']:.3f}s")
    print(f"Total time: {result['total_time']:.3f}s")
    print(f"Weak answer: {result.get('weak_answer', False)}")

    if result.get('rerank_enabled'):
        print("\nRerank diagnostics:")
        print(f"  enabled: {result.get('rerank_enabled')}")
        print(f"  retrieve_k: {result.get('retrieve_k')}")
        print(f"  before_top1: {result.get('rerank_before_top1') or 'n/a'}")
        print(f"  after_top1: {result.get('rerank_after_top1') or 'n/a'}")
        print(f"  top_scores: {result.get('rerank_top_scores') or []}")
        print(f"  context_length: {result.get('context_length')}")

    used = result.get('used_chunks') or result.get('context_chunks') or []
    if used:
        print("\nSources:")
        print(f"  (chunks used for generation: {len(used)})")
        for i, chunk in enumerate(used[:top_k], 1):
            source = chunk.get('metadata', {}).get('source_file', 'unknown')
            score = chunk.get('score', 0)
            chunk_id = chunk.get('metadata', {}).get('chunk_id', 'unknown')
            rerank_score = chunk.get('rerank_score')
            preview = (chunk.get('text') or '')[:120].replace('\n', ' ')
            if rerank_score is not None:
                print(f"  {i}. {source} (score: {score:.3f}, rerank: {rerank_score:.3f}) chunk_id={chunk_id} preview='{preview}...' ")
            else:
                print(f"  {i}. {source} (score: {score:.3f}) chunk_id={chunk_id} preview='{preview}...' ")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = _build_parser()
    args = parser.parse_args(argv)

    logger.info('Initializing RAG system')
    rag_system, audit_logger = _build_runtime(
        config_path=args.config,
        key_file=args.key_file,
        collection_name=args.collection_name,
        allow_empty_collection=args.allow_empty_collection,
    )
    logger.info('RAG system ready!')

    if args.question:
        process_question(
            rag_system,
            args.question,
            args.top_k,
            args.temperature,
            audit_logger,
        )
        return 0

    print("\n" + "=" * 50)
    print("Interactive RAG Q&A System")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 50 + "\n")

    while True:
        try:
            question = input("\nYour question: ").strip()
            if not question:
                continue
            if question.lower() in ['quit', 'exit', 'q']:
                print('Goodbye!')
                break
            process_question(
                rag_system,
                question,
                args.top_k,
                args.temperature,
                audit_logger,
            )
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f'Error: {e}')
            if audit_logger:
                audit_logger.log_error('question_processing', str(e))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
