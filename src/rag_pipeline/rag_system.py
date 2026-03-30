"""
RAG System
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import argparse
import logging
import os
import re
import time

import yaml

from src.audit import AuditLogger
from src.encryption import AESEncryption
from src.embedding import EmbeddingModel
from src.llm import OllamaClient
from src.rag_pipeline.rerank import LocalReranker
from src.retrieval import Retriever, VectorStore

logger = logging.getLogger(__name__)


class QueryType:
    TEMPORAL = "temporal"
    COMPARISON = "comparison"
    YES_NO = "yes_no"
    FACTOID = "factoid"
    OPEN = "open"


@dataclass(frozen=True)
class QueryInfo:
    query_type: str
    is_yes_no: bool


@dataclass(frozen=True)
class RuleResult:
    answer: str
    confidence: float
    rule_name: str


# =========================
# Query分类
# =========================

class QueryClassifier:

    def classify(self, question: str) -> QueryInfo:
        q = (question or "").lower()

        if any(k in q for k in ["before", "after", "earlier", "later"]):
            return QueryInfo(QueryType.COMPARISON, False)

        if any(k in q for k in ["when", "time", "date"]):
            return QueryInfo(QueryType.TEMPORAL, False)

        if q.startswith(("is ", "are ", "was ", "were ")):
            return QueryInfo(QueryType.YES_NO, True)

        return QueryInfo(QueryType.OPEN, False)


# =========================
# Temporal Rule（唯一保留规则）
# =========================

class TemporalComparisonRule:
    """
    泛化时间推理规则（论文核心：非硬编码）
    """

    name = "temporal_comparison"

    def apply(self, question: str, chunks: List[Dict]) -> Optional[RuleResult]:
        times = []

        for ch in chunks:
            text = ch.get("text", "")
            matches = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", text)
            for m in matches:
                try:
                    dt = datetime.strptime(m, "%Y-%m-%d %H:%M")
                    times.append(dt)
                except Exception:
                    continue

        if len(times) < 2:
            return None

        times.sort()

        if "difference" in (question or "").lower():
            delta = times[-1] - times[0]
            return RuleResult(
                answer=f"Time difference is {delta}",
                confidence=0.7,
                rule_name=self.name
            )

        return RuleResult(
            answer=f"Earliest time is {times[0]}",
            confidence=0.65,
            rule_name=self.name
        )


# =========================
# Rule Engine（只保留一个规则）
# =========================

class RuleEngine:
    def __init__(self):
        self.rule = TemporalComparisonRule()

    def run(self, question, chunks):
        return self.rule.apply(question, chunks)


# =========================
# Context Builder
# =========================

class ContextBuilder:
    """
    负责上下文构建（论文点：限制长度 + 提高相关性）
    """

    def __init__(self, max_chars: int = 2000, max_tokens: int = 650, min_rerank_score: float = 0.0):
        self.max_chars = int(max_chars)
        self.max_tokens = int(max_tokens)
        self.min_rerank_score = float(min_rerank_score)

    @staticmethod
    def _token_estimate(text: str) -> int:
        t = (text or "").strip()
        if not t:
            return 0
        return max(1, int(len(t) / 4) + int(len(t.split()) / 2))

    @staticmethod
    def _looks_temporal(question: str) -> bool:
        q = (question or "").lower()
        return any(k in q for k in ["before", "after", "earlier", "later", "when", "time", "date", "difference"])

    def build(self, question: str, chunks: List[Dict], top_k: int):
        if not chunks:
            return "", []

        # Prefer higher rerank_score chunks and suppress weak candidates.
        ranked = list(chunks)
        try:
            ranked.sort(key=lambda c: float(c.get("rerank_score", c.get("score", 0.0)) or 0.0), reverse=True)
        except Exception:
            pass

        # For temporal questions, bias toward chunks that actually contain timestamps or header-like evidence.
        temporal = self._looks_temporal(question)
        filtered = []
        for c in ranked:
            score = float(c.get("rerank_score", c.get("score", 0.0)) or 0.0)
            if score < self.min_rerank_score:
                continue
            text = (c.get("text") or "").strip()
            if not text:
                continue
            if temporal:
                tl = text.lower()
                has_time = bool(re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}", tl)) or any(h in tl for h in ("sent:", "date:", "time:", "from:", "to:", "subject:"))
                if not has_time:
                    # Keep a small amount of non-temporal support, but prefer evidence-rich chunks.
                    continue
            filtered.append({**c, "text": text})

        parts = []
        used = []
        total_chars = 0
        total_tokens = 0

        for idx, ch in enumerate(filtered[:top_k], 1):
            md = ch.get("metadata") if isinstance(ch.get("metadata"), dict) else {}
            src = md.get("source_file") or md.get("source") or md.get("source_file_name") or "unknown_source"
            cid = md.get("chunk_id")
            provenance = f"[source: {src}" + (f" chunk_id: {cid}" if cid is not None else "") + "]"
            block = f"[{idx}] {provenance}\n{ch.get('text', '')}".strip()

            block_tokens = self._token_estimate(block)
            if parts and (total_chars + len(block) > self.max_chars or total_tokens + block_tokens > self.max_tokens):
                break

            parts.append(block)
            used.append(ch)
            total_chars += len(block)
            total_tokens += block_tokens

        return "\n\n".join(parts), used


# =========================
# LLM 调用
# =========================

class LLMCaller:

    def __init__(self, client):
        self.client = client

    def generate(self, prompt: str):
        try:
            return self.client.generate(prompt=prompt)["response"]
        except Exception:
            return "Not found"


# =========================
# 主RAG系统
# =========================

class RAGSystem:

    def __init__(self, retriever, llm_client, prompt_template: Optional[str] = None, max_context_length: int = 2000, reranker=None):
        self.retriever = retriever
        self.classifier = QueryClassifier()
        self.context_builder = ContextBuilder()
        self.llm = LLMCaller(llm_client)
        self.rule_engine = RuleEngine()
        self.reranker = reranker
        self.prompt_template = prompt_template or self._default_prompt()
        self.max_context_length = max_context_length

    # =========================
    # Prompt设计（论文关键）
    # =========================
    def _default_prompt(self):
        return (
            "Answer ONLY using the provided context.\n"
            "If the answer is not in the context, say 'Not found'.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:\n"
        )

    def build_prompt(self, question, context):
        return self.prompt_template.format(context=context, question=question)

    @staticmethod
    def is_valid_answer(answer: str) -> bool:
        """Filter out weak or non-answers.

        Thesis note: free-form LLMs often produce vague fallback phrases when evidence
        is insufficient. We explicitly block these to keep the system behavior
        deterministic and easier to evaluate.
        """
        if not answer:
            return False
        a = answer.strip().lower()
        banned = ("not found", "i don't know", "i do not know", "cannot find", "no information")
        return not any(b in a for b in banned)

    # =========================
    # 核心流程（重点改造）
    # =========================
    def answer_question(self, question: str, top_k: int = 5, temperature: float = 0.7, max_tokens: Optional[int] = None):
        start = time.time()
        rerank_enabled = self.reranker is not None
        rerank_before_top1 = None
        rerank_after_top1 = None
        # Retrieve is widened to improve recall; rerank then restores precision.
        retrieve_k = max(top_k * 4, 20)
        context_length = 0
        rerank_top_scores: List[float] = []

        # 1️⃣ 检索（Recall first）
        try:
            chunks = self.retriever.retrieve(question, top_k=retrieve_k)
        except Exception as e:
            return {
                "answer": "An error occurred while searching for relevant information. Please try again later.",
                "path": "FAIL",
                "reasoning_path": "error",
                "time": time.time() - start,
                "total_time": time.time() - start,
                "retrieval_time": 0.0,
                "generation_time": 0.0,
                "num_chunks_retrieved": 0,
                "used_chunks": [],
                "context_chunks": [],
                "confidence": 0.0,
                "error": str(e),
                "rerank_enabled": rerank_enabled,
                "rerank_before_top1": rerank_before_top1,
                "rerank_after_top1": rerank_after_top1,
                "rerank_top_scores": rerank_top_scores,
                "context_length": context_length,
                "retrieve_k": retrieve_k,
            }
        retrieval_time = time.time() - start

        # 1.5️⃣ Rerank（Precision first）
        # Why rerank: vector retrieval maximizes recall, but the initial ordering is not
        # always evidence-precise enough for generation. Reranking re-orders candidates so
        # the context builder sees the most relevant chunks first.
        if self.reranker is not None:
            try:
                rerank_before_top1 = chunks[0].get("text") if chunks else None
                chunks = self.reranker.rerank(question, chunks, top_k=retrieve_k)
                rerank_after_top1 = chunks[0].get("text") if chunks else None
            except Exception as e:
                logger.debug("Reranker failed, falling back to raw retrieval order: %s", e)

        rerank_top_scores = [float(ch.get("rerank_score", ch.get("score", 0.0)) or 0.0) for ch in (chunks or [])[:3]]

        # 2️⃣ 构建上下文
        # ContextBuilder is responsible for token/character budgeting and for preferring
        # higher-quality reranked chunks. This prevents noisy evidence from overflowing
        # the prompt and improves answer stability.
        context, used_chunks = self.context_builder.build(question, chunks, top_k)
        context_length = len(context)

        # 3️⃣ 构建prompt
        prompt = self.build_prompt(question, context)

        # 4️⃣ LLM生成（主路径）
        gen_start = time.time()
        answer = self.llm.generate(prompt)
        generation_time = time.time() - gen_start

        # =========================
        # ❗ 核心：RAG优先
        # =========================
        if self.is_valid_answer(answer):
            return {
                "answer": answer,
                "path": "RAG",
                "reasoning_path": "llm",
                "time": time.time() - start,
                "total_time": time.time() - start,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "num_chunks_retrieved": len(chunks),
                "used_chunks": used_chunks,
                "context_chunks": chunks,
                "confidence": 0.55,
                "rerank_enabled": rerank_enabled,
                "rerank_before_top1": rerank_before_top1,
                "rerank_after_top1": rerank_after_top1,
                "rerank_top_scores": rerank_top_scores,
                "context_length": context_length,
                "retrieve_k": retrieve_k,
                "weak_answer": False,
            }

        # =========================
        # ❗ fallback：规则
        # =========================
        rule_result = self.rule_engine.run(question, chunks)

        if rule_result:
            return {
                "answer": rule_result.answer,
                "path": f"RULE:{rule_result.rule_name}",
                "reasoning_path": f"fallback_rule:{rule_result.rule_name}",
                "time": time.time() - start,
                "total_time": time.time() - start,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "num_chunks_retrieved": len(chunks),
                "used_chunks": used_chunks,
                "context_chunks": chunks,
                "confidence": rule_result.confidence,
                "rerank_enabled": rerank_enabled,
                "rerank_before_top1": rerank_before_top1,
                "rerank_after_top1": rerank_after_top1,
                "rerank_top_scores": rerank_top_scores,
                "context_length": context_length,
                "retrieve_k": retrieve_k,
                "weak_answer": True,
            }

        return {
            "answer": "Not found",
            "path": "FAIL",
            "reasoning_path": "llm",
            "time": time.time() - start,
            "total_time": time.time() - start,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "num_chunks_retrieved": len(chunks),
            "used_chunks": used_chunks,
            "context_chunks": chunks,
            "confidence": 0.0,
            "rerank_enabled": rerank_enabled,
            "rerank_before_top1": rerank_before_top1,
            "rerank_after_top1": rerank_after_top1,
            "rerank_top_scores": rerank_top_scores,
            "context_length": context_length,
            "retrieve_k": retrieve_k,
            "weak_answer": True,
        }


def _build_parser():
    parser = argparse.ArgumentParser(description='Run RAG system for Q&A')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--key_file', type=str, default='encryption.key', help='Path to encryption key file')
    parser.add_argument('--question', type=str, help='Single question to answer (if not provided, runs interactive mode)')
    parser.add_argument('--top_k', type=int, default=5, help='Number of chunks to retrieve')
    parser.add_argument('--temperature', type=float, default=0.7, help='LLM temperature')
    parser.add_argument('--exact_extract', action='store_true', help='Prefer deterministic exact extraction for date/time style questions')
    parser.add_argument('--collection_name', type=str, default=None, help='Override Qdrant collection name for this run')
    return parser


def _build_runtime(config_path: str, key_file: str, collection_name: Optional[str] = None):
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
    if info.get('points_count', 0) == 0:
        raise RuntimeError('Vector database is empty. Please run ingest_documents.py first')

    retriever = Retriever(embedding_model, vector_store, encryption)
    llm_client = OllamaClient(base_url=config['llm']['base_url'], model_name=config['llm']['model_name'])
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
        prompt_template=config['rag']['prompt_template'],
        max_context_length=config['rag']['max_context_length'],
        reranker=reranker
    )
    return rag_system, audit_logger


def process_question(rag_system, question, top_k, temperature, audit_logger=None, exact_extract=False):
    del exact_extract
    print(f"\nQuestion: {question}")
    print("-" * 50)

    if audit_logger:
        audit_logger.log_query(question)

    result = rag_system.answer_question(question=question, top_k=top_k, temperature=temperature)

    if audit_logger:
        model_obj = getattr(getattr(rag_system, 'llm', None), 'client', None)
        model_name = getattr(model_obj, 'model_name', 'unknown-model')
        audit_logger.log_model_invocation(model_name=model_name, inference_time=result['generation_time'])

    print(f"\nAnswer: {result['answer']}")
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
        try:
            nchunks = len(used)
        except Exception:
            nchunks = 0
        print(f"  (chunks used for generation: {nchunks})")
        for i, chunk in enumerate(used[:top_k], 1):
            source = chunk.get('metadata', {}).get('source_file', 'unknown')
            score = chunk.get('score', 0)
            chunk_id = chunk.get('metadata', {}).get('chunk_id', 'unknown')
            rerank_score = chunk.get('rerank_score')
            rerank_reason = chunk.get('rerank_reason')
            preview = (chunk.get('text') or '')[:120].replace('\n', ' ')
            if rerank_score is not None:
                print(f"  {i}. {source} (score: {score:.3f}, rerank: {rerank_score:.3f}) chunk_id={chunk_id} preview='{preview}...' ")
                print(f"     rerank_reason: {rerank_reason}")
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
    )
    logger.info('RAG system ready!')

    if args.question:
        process_question(rag_system, args.question, args.top_k, args.temperature, audit_logger, args.exact_extract)
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
            process_question(rag_system, question, args.top_k, args.temperature, audit_logger, args.exact_extract)
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
