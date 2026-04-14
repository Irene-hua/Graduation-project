import pytest


def _parse_compact_datetime(text: str):
    """Parse compact 'YYYY-MM-DD 7 PM' or 'YYYY-MM-DD 19:00' like outputs.

    We keep this lenient because different models may format time slightly differently.
    """
    import re

    t = (text or "").strip()
    # accept YYYY-MM-DD 7 PM / 7PM / 19:00
    m = re.search(r"(20\d{2}-\d{2}-\d{2})\s+([0-9]{1,2})(?::([0-9]{2}))?\s*(AM|PM)?", t, re.IGNORECASE)
    if not m:
        return None
    date = m.group(1)
    hour = int(m.group(2))
    minute = int(m.group(3) or "0")
    ampm = (m.group(4) or "").upper()

    if ampm == "PM" and hour != 12:
        hour += 12
    if ampm == "AM" and hour == 12:
        hour = 0

    return f"{date} {hour:02d}:{minute:02d}"


@pytest.mark.unit
def test_expected_time_inference_case_should_be_answerable_by_prompt():
    """This is a contract-style test showing the *intended* behavior.

    We don't call Ollama here. Instead, we make sure the prompt contains the notes
    required to resolve 'tomorrow' relative to an explicit timestamp.

    The real correctness is validated via manual/CLI or an optional integration test.
    """

    from src.rag_pipeline.prompts import CONTROLLED_REASONING_RAG_PROMPT

    # minimal context needed for the model to infer tomorrow
    context = (
        "2026-01-21 09:00\n"
        "Li Hua: Let's watch Overwatch 3 tomorrow at 7 PM."
    )
    question = "What time does Li Hua watch the movie Overwatch 3?"

    prompt = CONTROLLED_REASONING_RAG_PROMPT.format(context=context, question=question)

    assert "tomorrow" in prompt.lower()
    assert "relative" in prompt.lower()

    # sanity-check our parser expectation for future integration assertions
    assert _parse_compact_datetime("2026-01-22 7 PM") == "2026-01-22 19:00"

