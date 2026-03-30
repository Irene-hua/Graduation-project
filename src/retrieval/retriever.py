"""
Retriever
Combines embedding, vector search, and decryption for retrieval.

This retriever is designed to be robust and *generic*:
- Avoids hardcoded special entities/files.
- Uses lightweight query classification to enable targeted fallbacks.
- Supports an optional full-scan fallback (off by default) for offline evaluation/debug.
"""

from typing import List, Dict, Optional, Set
import numpy as np
import logging
import re

logger = logging.getLogger(__name__)


class Retriever:
    """High-level retriever combining embedding, search, and decryption"""

    # Email header detection
    _HEADER_PATTERNS = (
        re.compile(r"^sent:\s*", re.IGNORECASE),
        re.compile(r"^date:\s*", re.IGNORECASE),
        re.compile(r"^from:\s*", re.IGNORECASE),
        re.compile(r"^to:\s*", re.IGNORECASE),
        re.compile(r"^subject:\s*", re.IGNORECASE),
    )

    def __init__(self, embedding_model, vector_store, encryption):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.encryption = encryption

    # ------------------------- Query understanding -------------------------

    def _query_tokens(self, query: str) -> Set[str]:
        return set(re.findall(r"\w+", (query or "").lower()))

    def _looks_like_time_question(self, query: str) -> bool:
        q = (query or "").lower()
        return any(k in q for k in [
            "date and time",
            "what is the date",
            "what is the time",
            "timestamp",
            "when",
            "responded",
            "replied",
            "sent",
        ])

    def _looks_like_email_thread(self, query: str) -> bool:
        q = (query or "").lower()
        return any(k in q for k in ["email", "subject", "sender", "recipient", "re:", "fw:"])

    # ------------------------- Payload/plaintext helpers -------------------------

    def _extract_plaintext_from_payload(self, payload: Dict) -> Optional[str]:
        """Try plaintext fields first; if encrypted, decrypt."""
        pd = self.vector_store._normalize_payload(payload)

        for key in ("text", "plaintext", "content", "document", "source_text"):
            if pd.get(key):
                return str(pd.get(key))

        ct = pd.get("ciphertext") or pd.get("ct")
        nonce = pd.get("nonce") or pd.get("n")
        if ct and nonce:
            try:
                return self.encryption.decrypt(ct, nonce)
            except Exception:
                return None
        return None

    def _looks_like_encrypted_payload(self, payload: Dict) -> bool:
        """Heuristic check for whether a payload really contains encrypted content."""
        pd = self.vector_store._normalize_payload(payload)
        ct = pd.get("ciphertext") or pd.get("ct")
        nonce = pd.get("nonce") or pd.get("n")

        if not ct or not nonce:
            return False

        # Only treat string-like values as real encrypted payloads.
        return isinstance(ct, (str, bytes)) and isinstance(nonce, (str, bytes))

    def _safe_decrypt_payload(self, payload: Dict) -> Optional[str]:
        """Decrypt only when the payload actually looks encrypted; otherwise use plaintext fields."""
        pd = self.vector_store._normalize_payload(payload)

        for key in ("text", "plaintext", "content", "document", "source_text"):
            if pd.get(key):
                return str(pd.get(key))

        if not self._looks_like_encrypted_payload(pd):
            return None

        ct = pd.get("ciphertext") or pd.get("ct")
        nonce = pd.get("nonce") or pd.get("n")
        try:
            return self.encryption.decrypt(ct, nonce)
        except Exception:
            return None

    def _has_email_headers(self, text: str) -> bool:
        if not text:
            return False
        for line in text.splitlines()[:40]:
            s = line.strip()
            if not s:
                continue
            if any(pat.match(s) for pat in self._HEADER_PATTERNS):
                return True
        return False

    # ------------------------- Targeted fallback for time/header queries -------------------------

    def _header_boost_fallback(self, query: str, top_k: int) -> List[Dict]:
        """Boost recall for timestamp/header questions by scanning a limited number of payloads.

        This is intentionally capped to avoid an O(N) full scan by default.
        """
        if not (self._looks_like_time_question(query) or self._looks_like_email_thread(query)):
            return []

        try:
            # Scroll a limited number of points.
            recs = self.vector_store.get_all_points(batch_size=800, with_payload=True, with_vectors=False)
        except Exception as e:
            logger.debug(f"Header boost scan failed: {e}")
            return []

        q_tokens = self._query_tokens(query)
        candidates: List[Dict] = []

        for rec in recs:
            try:
                payload = getattr(rec, "payload", None) or (rec.payload if hasattr(rec, "payload") else {})
                text = self._extract_plaintext_from_payload(payload) or ""
                if not text:
                    continue

                # Only keep chunks that look like email headers to avoid noise.
                if not self._has_email_headers(text):
                    continue

                # Score by lexical overlap + strong boost if Sent header exists.
                tlow = text.lower()
                tokens = set(re.findall(r"\w+", tlow))
                overlap = 0.0
                if q_tokens and tokens:
                    overlap = len(q_tokens & tokens) / max(1, len(q_tokens))

                if "sent:" in tlow or "date:" in tlow:
                    overlap += 1.0

                if overlap <= 0.0:
                    continue

                pd = self.vector_store._normalize_payload(payload)
                candidates.append({
                    "id": getattr(rec, "id", None),
                    "score": float(overlap),
                    "text": text,
                    "metadata": {k: v for k, v in pd.items() if k not in ["ciphertext", "nonce", "ct", "n"]},
                })

                if len(candidates) >= max(30, top_k * 10):
                    # keep bounded
                    break
            except Exception:
                continue

        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return candidates[: max(3, top_k)]

    def _contains_headers_in_results(self, results: List[Dict]) -> bool:
        """Check whether already-decrypted results contain email header lines."""
        for r in results or []:
            try:
                txt = r.get("text") or ""
                if self._has_email_headers(str(txt)):
                    return True
            except Exception:
                continue
        return False

    # ------------------------- Main retrieve -------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        return_encrypted: bool = False,
        allow_local_fallback: bool = False,
    ) -> List[Dict]:
        """Retrieve and decrypt relevant chunks for a query."""

        # 1) Encode query
        logger.info(f"Encoding query: {query[:50]}...")
        query_vector = self.embedding_model.encode(query)
        query_tokens = self._query_tokens(query)

        # 2) Vector search (semantic-first)
        search_results: List[Dict] = []
        try:
            logger.info(f"Searching for top-{top_k} similar chunks...")
            search_results = self.vector_store.search(query_vector, top_k=top_k)
            logger.info("Search returned %d results", len(search_results) if search_results else 0)
        except Exception as e:
            logger.debug(f"Vector search failed: {e}")
            search_results = []

        # 3) Decrypt / normalize vector search results
        decrypted_results: List[Dict] = []
        logger.info(f"Decrypting {len(search_results)} chunks...")

        for result in search_results or []:
            try:
                ciphertext = result.get("ciphertext")
                nonce = result.get("nonce")

                metadata = result.get("metadata") if result.get("metadata") is not None else {}
                if not isinstance(metadata, dict):
                    try:
                        metadata = dict(metadata)
                    except Exception:
                        metadata = {}

                # Some results may nest encrypted data in metadata
                if (ciphertext is None or nonce is None) and isinstance(metadata, dict):
                    ciphertext = ciphertext or metadata.get("ciphertext") or metadata.get("ct")
                    nonce = nonce or metadata.get("nonce") or metadata.get("n")

                plaintext: Optional[str] = None
                if ciphertext is not None and nonce is not None:
                    if isinstance(ciphertext, (str, bytes)) and isinstance(nonce, (str, bytes)):
                        plaintext = self.encryption.decrypt(ciphertext, nonce)
                    else:
                        plaintext = None
                if plaintext is None:
                    for key in ("text", "plaintext", "content", "document", "source_text"):
                        if key in metadata and metadata[key]:
                            plaintext = str(metadata[key])
                            break

                # If encrypted payload is malformed, skip it silently instead of spamming errors.
                if plaintext is None and (ciphertext is not None or nonce is not None):
                    logger.debug(
                        "Skipping malformed encrypted payload for chunk %s (ciphertext=%s, nonce=%s)",
                        result.get('id', None),
                        type(ciphertext).__name__,
                        type(nonce).__name__,
                    )
                    continue

                if plaintext is None:
                    continue

                decrypted_result = {
                    "text": plaintext,
                    "score": result.get("score"),
                    "metadata": metadata,
                }

                if return_encrypted:
                    decrypted_result["ciphertext"] = ciphertext
                    decrypted_result["nonce"] = nonce

                decrypted_results.append(decrypted_result)

            except Exception as e:
                logger.debug(f"Skipping chunk {result.get('id', None)} due to retrieval normalization error: {e}")
                continue

        # 4) Two-stage header boost: only for time/email queries when semantic results don't contain headers
        if self._looks_like_time_question(query) or self._looks_like_email_thread(query):
            if not self._contains_headers_in_results(decrypted_results):
                boosted = self._header_boost_fallback(query, top_k=top_k)
                if boosted:
                    logger.info(
                        "Two-stage header boost appended %d candidate(s) (no headers in semantic top_k)",
                        len(boosted),
                    )
                    # Merge (keep semantic first, then boosted; avoid duplicates by text key)
                    seen = set()
                    merged: List[Dict] = []
                    for it in decrypted_results + boosted:
                        txt = (it.get("text") or "").strip().lower()
                        k = re.sub(r"\s+", " ", txt)[:300]
                        if not k or k in seen:
                            continue
                        seen.add(k)
                        merged.append(it)
                        if len(merged) >= top_k:
                            break
                    decrypted_results = merged

        logger.info(f"Successfully retrieved and decrypted {len(decrypted_results)} chunks")

        # 5) Optional full local fallback (expensive) for offline debugging
        if (not decrypted_results) and allow_local_fallback:
            try:
                logger.info("Using local full-scan fallback (allow_local_fallback=True)")
                recs = self.vector_store.get_all_points(batch_size=1000, with_payload=True, with_vectors=False)

                lex_items: List[Dict] = []
                for rec in recs:
                    try:
                        payload = getattr(rec, "payload", None) or (rec.payload if hasattr(rec, "payload") else {})
                        text = self._extract_plaintext_from_payload(payload)
                        if not text:
                            continue
                        tokens = set(re.findall(r"\w+", text.lower()))
                        overlap = 0.0
                        if query_tokens and tokens:
                            overlap = len(query_tokens & tokens) / max(1, len(queryTokens))
                        if overlap <= 0.0:
                            continue

                        pd = self.vector_store._normalize_payload(payload)
                        lex_items.append({
                            "id": getattr(rec, "id", None),
                            "score": overlap,
                            "text": text,
                            "metadata": {k: v for k, v in pd.items() if k not in ["ciphertext", "nonce", "ct", "n"]},
                        })
                    except Exception:
                        continue

                lex_items.sort(key=lambda x: x.get("score", 0.0), reverse=True)
                if lex_items:
                    logger.info("Local lexical fallback produced %d candidates", len(lex_items[:top_k]))
                    return lex_items[:top_k]
            except Exception as e:
                logger.debug(f"Local full-scan fallback failed: {e}")

        return decrypted_results

    def retrieve_batch(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        results = []
        for query in queries:
            results.append(self.retrieve(query, top_k=top_k))
        return results

    def evaluate_retrieval(self, query: str, ground_truth_ids: List[str], top_k: int = 5) -> Dict:
        results = self.retrieve(query, top_k=top_k)
        retrieved_ids = [r.get("metadata", {}).get("chunk_id", "") for r in results]
        relevant_retrieved = len(set(retrieved_ids) & set(ground_truth_ids))

        precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0
        recall = relevant_retrieved / len(ground_truth_ids) if ground_truth_ids else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "retrieved_count": len(retrieved_ids),
            "relevant_count": len(ground_truth_ids),
            "relevant_retrieved": relevant_retrieved,
        }
