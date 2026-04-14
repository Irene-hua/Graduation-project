import re

import pytest


@pytest.mark.unit
def test_controlled_reasoning_prompt_includes_reasoning_instructions():
    """Prompt contract for temporal reasoning:
    - must be context-only
    - must require reasoning when needed
    - must explicitly include tomorrow rule + timestamp reference
    - must enforce strict Answer/Explanation output format
    - must allow partial answers
    """

    from src.rag_pipeline.prompts import CONTROLLED_REASONING_RAG_PROMPT

    p = CONTROLLED_REASONING_RAG_PROMPT

    assert "answer based ONLY on the provided context" in p
    assert "REQUIRED to perform reasoning" in p

    # Must explicitly mention timestamp + tomorrow rule
    assert "Identify any date or timestamp" in p
    assert "\"tomorrow\" = next day" in p
    assert "timestamp" in p.lower()

    # Must not ignore time expressions
    assert "MUST NOT ignore \"tomorrow\"" in p

    # Partial answers allowed
    assert "If partial information exists" in p

    # Output format contract
    assert "OUTPUT FORMAT (STRICT)" in p
    assert "Answer: <final answer>" in p
    assert "Explanation: <one sentence" in p

    # Placeholders
    assert "{context}" in p
    assert "{question}" in p


@pytest.mark.unit
def test_prompt_has_single_context_and_question_blocks():
    from src.rag_pipeline.prompts import CONTROLLED_REASONING_RAG_PROMPT

    p = CONTROLLED_REASONING_RAG_PROMPT
    assert len(re.findall(r"\bContext:\s*\{context\}", p)) == 1
    assert len(re.findall(r"\bQuestion:\s*\{question\}", p)) == 1
