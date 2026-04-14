"""Optional integration test for forced step-by-step reasoning.

This test is skipped by default because it depends on:
- a running Ollama server
- the target Qdrant collection having the relevant evidence

Enable by setting:
  RUN_OLLAMA_INTEGRATION_TESTS=1

and ensure ollama is available.
"""

from __future__ import annotations

import os
import re

import pytest


def _has_steps(text: str) -> bool:
    t = (text or "").lower()
    return "step 1" in t and "step 2" in t and "step 3" in t


@pytest.mark.skipif(os.getenv("RUN_OLLAMA_INTEGRATION_TESTS") != "1", reason="integration test requires Ollama + dataset")
def test_overwatch3_time_reasoning_has_steps_and_final_datetime():
    from src.rag_pipeline.rag_system import _build_runtime

    rag, _ = _build_runtime(
        config_path="config/config.yaml",
        key_file="encryption.key",
        collection_name=os.getenv("INTEGRATION_COLLECTION", "encrypted_documents_lihua"),
        allow_empty_collection=False,
    )

    q = 'What time does Li Hua watch the movie "Overwatch 3"?'
    res = rag.answer_question(q, top_k=10, temperature=0.0, max_tokens=256)
    ans = (res.get("answer") or "").strip()

    assert _has_steps(ans), ans
    assert "i don't know" not in ans.lower(), ans

    # Very lenient: any YYYY-MM-DD and time marker in Step 3 section.
    m = re.search(r"step\s*3\s*:(.*)$", ans, flags=re.IGNORECASE | re.DOTALL)
    step3 = (m.group(1) if m else ans).strip()
    assert re.search(r"20\d{2}-\d{2}-\d{2}", step3), step3
    assert re.search(r"\b(\d{1,2}(:\d{2})?\s*(AM|PM)?|\d{2}:\d{2})\b", step3, flags=re.IGNORECASE), step3

