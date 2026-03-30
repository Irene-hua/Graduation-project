"""
Hybrid local reranker for thesis-grade two-stage retrieval.

Design goals:
- Keep everything local: use Hugging Face / sentence-transformers CrossEncoder if available.
- If CrossEncoder cannot be loaded, gracefully fall back to the lightweight lexical LocalReranker.
- Preserve modularity: a class-based reranker that can be injected into RAGSystem.
- Improve precision after widening retrieve_k for recall.

Why this matters in a RAG pipeline:
1) Retrieve expands recall and surfaces more candidate evidence.
2) Rerank improves precision by re-ordering those candidates with a stronger matching model.
3) ContextBuilder then selects a compact, high-quality subset for generation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import logging
import re

logger = logging.getLogger(__name__)


class HybridReranker:
    """Two-mode local reranker.

    Mode A: CrossEncoder-based reranking (preferred)
    Mode B: Lightweight lexical reranking (fallback)

    This is intentionally a class, not a function, so it can be configured,
    tested, and injected without changing the overall modular architecture.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_candidates: int = 20, min_score: float = 0.0):
        self.model_name = model_name
        self.max_candidates = int(max_candidates)
        self.min_score = float(min_score)
        self._cross_encoder = None
        self._fallback = None
        self._mode = "local"

        # Try to load a fully local CrossEncoder first.
        # If this fails, we fall back to the deterministic local reranker.
        try:
            from sentence_transformers import CrossEncoder  # local dependency, no cloud API
            self._cross_encoder = CrossEncoder(self.model_name)
            self._mode = "cross_encoder"
            logger.info("HybridReranker loaded CrossEncoder model: %s", self.model_name)
        except Exception as e:
            logger.warning("CrossEncoder unavailable, falling back to LocalReranker: %s", e)
            from .rerank import LocalReranker
            self._fallback = LocalReranker(max_candidates=max_candidates, min_score=min_score)
            self._mode = "local"

    @property
    def mode(self) -> str:
        return self._mode

    @staticmethod
    def _extract_text(chunk: Dict) -> str:
        return (chunk.get("text") or "").strip()

    @staticmethod
    def _temporal_prior(question: str, text: str) -> float:
        """Add a strong but still bounded prior for time-oriented questions.

        Thesis rationale: temporal queries often depend on evidence containing explicit
        timestamps or chronology markers. A small prior helps such chunks move upward
        before context construction without changing the pipeline structure.
        """
        q = (question or "").lower()
        t = (text or "").lower()
        if not any(kw in q for kw in ("before", "after", "earlier", "later", "when", "time", "date", "difference")):
            return 0.0

        prior = 0.0
        if re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}", t):
            prior += 0.20
        if any(h in t for h in ("sent:", "date:", "from:", "to:", "subject:", "time:")):
            prior += 0.10
        if "time:" in t or re.search(r"\b\d{1,2}(:\d{2})?\s?(am|pm)\b", t):
            prior += 0.08
        return prior

    def _rerank_with_cross_encoder(self, question: str, chunks: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        if not chunks:
            return []

        # The CrossEncoder scores (query, passage) pairs directly.
        pairs = [(question, self._extract_text(ch)) for ch in chunks[: self.max_candidates]]
        scores = self._cross_encoder.predict(pairs)

        scored_chunks: List[Dict] = []
        for chunk, score in zip(chunks[: self.max_candidates], scores):
            enriched = dict(chunk)
            final_score = float(score) + self._temporal_prior(question, self._extract_text(chunk))
            enriched["rerank_score"] = final_score
            enriched["rerank_source"] = "cross_encoder"
            enriched["rerank_reason"] = f"cross_encoder:{self.model_name}"
            scored_chunks.append(enriched)

        scored_chunks.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        scored_chunks = [c for c in scored_chunks if c.get("rerank_score", 0.0) >= self.min_score]
        return scored_chunks[:top_k] if top_k is not None else scored_chunks

    def rerank(self, question: str, chunks: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        if self._cross_encoder is not None:
            try:
                return self._rerank_with_cross_encoder(question, chunks, top_k=top_k)
            except Exception as e:
                logger.warning("CrossEncoder rerank failed, switching to local fallback: %s", e)
                # If the preferred mode fails at runtime, fall back to the deterministic local scorer.
                from .rerank import LocalReranker
                self._fallback = self._fallback or LocalReranker(max_candidates=self.max_candidates, min_score=self.min_score)
                self._cross_encoder = None
                self._mode = "local"

        if self._fallback is None:
            from .rerank import LocalReranker
            self._fallback = LocalReranker(max_candidates=self.max_candidates, min_score=self.min_score)
        return self._fallback.rerank(question, chunks, top_k=top_k)
