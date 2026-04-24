"""
Local Rerank module.

Design goals:
- Keep the pipeline fully local (no cloud API).
- Preserve modularity: reranking is a standalone class, not a function.
- Improve retrieval precision by reordering the initial top-k candidates before context construction.
- Stay lightweight and deterministic enough for local batch testing and thesis reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import logging
import re

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankScore:
    score: float
    reason: str


class LocalReranker:
    """A lightweight, fully local reranker based on lexical overlap and query intent signals.

    Rationale:
    - We do not introduce cloud dependencies.
    - We keep the implementation deterministic and easy to explain in a thesis.
    - We improve over pure vector similarity by rescoring candidates against the query.
    - We add tiny metadata priors so structured chunks (e.g. temporal/email-like chunks)
      can win ties without changing the modular pipeline.
    """

    _STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "if", "then", "than", "to", "of",
        "in", "on", "for", "with", "by", "is", "are", "was", "were", "be", "been",
        "being", "it", "this", "that", "these", "those", "as", "at", "from", "into",
        "about", "after", "before", "when", "where", "who", "what", "which", "why",
        "how", "do", "does", "did", "can", "could", "should", "would", "may", "might",
    }

    def __init__(self, max_candidates: int = 20, min_score: float = 0.0):
        self.max_candidates = int(max_candidates)
        self.min_score = float(min_score)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            t for t in re.findall(r"\w+", (text or "").lower())
            if t and t not in LocalReranker._STOPWORDS
        }

    @staticmethod
    def _time_signals(question: str) -> set[str]:
        q = (question or "").lower()
        signals = set()
        for kw in ("before", "after", "earlier", "later", "when", "time", "date", "difference"):
            if kw in q:
                signals.add(kw)
        return signals

    @staticmethod
    def _metadata_prior(chunk: Dict) -> Tuple[float, str]:
        """Small deterministic priors to help tie-breaking without changing retrieval semantics."""
        md = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        text = (chunk.get("text") or "").lower()

        prior = 0.0
        reasons: List[str] = []

        # Slightly prefer chunks that have a source file and chunk id because they tend to be cleaner payloads.
        if md.get("source_file"):
            prior += 0.01
            reasons.append("source_file")
        if md.get("chunk_id") is not None:
            prior += 0.01
            reasons.append("chunk_id")

        # Temporal/email-like content is especially useful for the thesis datasets.
        if re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}", text):
            prior += 0.05
            reasons.append("timestamp")
        if any(h in text for h in ("sent:", "date:", "from:", "to:", "subject:")):
            prior += 0.05
            reasons.append("header")

        reason = "+".join(reasons) if reasons else "no_prior"
        return prior, reason

    def score(self, question: str, chunk: Dict) -> RerankScore:
        text = (chunk.get("text") or "")
        q_tokens = self._tokens(question)
        c_tokens = self._tokens(text)

        if not q_tokens or not c_tokens:
            prior, prior_reason = self._metadata_prior(chunk)
            score = prior
            return RerankScore(score=score, reason=f"empty_query_or_chunk;prior={prior_reason}")

        overlap = len(q_tokens & c_tokens)
        base = overlap / max(1, len(q_tokens))

        # Small boost for exact token matches to improve ranking precision.
        t_lower = text.lower()
        phrase_boost = 0.0
        for term in list(q_tokens)[:8]:
            if term and term in t_lower:
                phrase_boost += 0.02

        # Temporal queries benefit from time-like content being ranked higher.
        temporal_boost = 0.0
        signals = self._time_signals(question)
        if signals and re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}", text):
            temporal_boost = 0.15

        prior, prior_reason = self._metadata_prior(chunk)
        score = base + phrase_boost + temporal_boost + prior
        return RerankScore(
            score=score,
            reason=(
                f"overlap={overlap},phrase_boost={phrase_boost:.2f},"
                f"temporal_boost={temporal_boost:.2f},prior={prior_reason}"
            ),
        )

    def rerank(self, question: str, chunks: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        """Return chunks re-ordered by local rerank score.

        If top_k is provided, the output is trimmed after reranking.
        """
        if not chunks:
            return []

        scored: List[Tuple[float, Dict]] = []
        for chunk in chunks[: self.max_candidates]:
            s = self.score(question, chunk)
            enriched = dict(chunk)
            enriched["rerank_score"] = s.score
            enriched["rerank_reason"] = s.reason
            scored.append((s.score, enriched))

        scored.sort(key=lambda item: item[0], reverse=True)
        reranked = [item[1] for item in scored if item[0] >= self.min_score]

        if top_k is not None:
            return reranked[:top_k]
        return reranked


# --- Begin merged HybridReranker (from hybrid_rerank.py) ---

from typing import Dict, List, Optional
import re


class HybridReranker:
    """Two-mode local reranker.

    Mode A: CrossEncoder-based reranking (preferred)
    Mode B: Lightweight lexical reranking (fallback)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_candidates: int = 20, min_score: float = 0.0):
        self.model_name = model_name
        self.max_candidates = int(max_candidates)
        self.min_score = float(min_score)
        self._cross_encoder = None
        self._fallback = None
        self._mode = "local"

        try:
            from sentence_transformers import CrossEncoder  # local dependency, no cloud API
            self._cross_encoder = CrossEncoder(self.model_name)
            self._mode = "cross_encoder"
            logger.info("HybridReranker loaded CrossEncoder model: %s", self.model_name)
        except Exception as e:
            logger.warning("CrossEncoder unavailable, falling back to LocalReranker: %s", e)
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
                from .rerank import LocalReranker
                self._fallback = self._fallback or LocalReranker(max_candidates=self.max_candidates, min_score=self.min_score)
                self._cross_encoder = None
                self._mode = "local"

        if self._fallback is None:
            from .rerank import LocalReranker
            self._fallback = LocalReranker(max_candidates=self.max_candidates, min_score=self.min_score)
        return self._fallback.rerank(question, chunks, top_k=top_k)
