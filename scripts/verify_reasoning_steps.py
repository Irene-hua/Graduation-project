"""Integration verifier for forced step-by-step reasoning outputs.

Purpose
-------
This script provides an *automatic* pass/fail check for the compact
Answer/Explanation format and a minimal temporal reasoning expectation.

It is designed to validate that the system:
- doesn't bypass reasoning
- doesn't answer "I don't know" when the answer is derivable
- keeps the Step format stable

Usage (PowerShell)
------------------
python -m scripts.verify_reasoning_steps --collection_name encrypted_documents_lihua --question "What time does Li Hua watch the movie Overwatch 3?"

Exit code:
- 0: PASS
- 2: FAIL (format/derivation)
- 3: RUNTIME ERROR
"""

from __future__ import annotations

import argparse
import re


def _has_steps(text: str) -> bool:
    t = (text or "").lower()
    return "step 1" in t and "step 2" in t and "step 3" in t


def _has_answer_explanation(text: str) -> bool:
    t = (text or "").strip()
    tl = t.lower()
    return tl.startswith("answer:") and ("\nexplanation:" in tl)


def _extract_answer_line(text: str) -> str:
    if not text:
        return ""
    for line in text.splitlines():
        if line.strip().lower().startswith("answer:"):
            return line.strip()
    return ""


def _extract_explanation_line(text: str) -> str:
    if not text:
        return ""
    for line in text.splitlines():
        if line.strip().lower().startswith("explanation:"):
            return line.strip()
    return ""


def _is_one_sentence_explanation(line: str) -> bool:
    """Heuristic: ensure it's short and not multi-sentence."""
    if not line:
        return False
    # remove prefix
    body = line.split(":", 1)[-1].strip()
    if not body:
        return False
    # Allow one sentence that may end with '.'; reject multiple sentence delimiters.
    # Keep it lenient for Chinese/English.
    if body.count(".") > 1:
        return False
    if body.count("?") > 0 and body.count("?") > 1:
        return False
    if body.count("!") > 0 and body.count("!") > 1:
        return False
    return len(body) <= 220


def _extract_final_answer_block(text: str) -> str:
    """Return the answer content for datetime checks (best-effort)."""
    line = _extract_answer_line(text)
    return line


def _contains_datetime_like(text: str) -> bool:
    # Accept either '2026-01-22 7 PM' or '2026-01-22 19:00' etc.
    if not text:
        return False
    t = text.strip()
    return bool(re.search(r"20\d{2}-\d{2}-\d{2}.*\b(\d{1,2}(:\d{2})?\s*(AM|PM)?|\d{2}:\d{2})\b", t, re.IGNORECASE))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify forced step-by-step reasoning output")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--key_file", default="encryption.key")
    parser.add_argument("--collection_name", default=None)
    parser.add_argument(
        "--question",
        default="What time does Li Hua watch the movie Overwatch 3?",
        help="Question to ask (default is Overwatch 3 time reasoning)",
    )
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args(argv)

    try:
        from src.rag_pipeline.rag_system import _build_runtime

        rag, _ = _build_runtime(
            config_path=args.config,
            key_file=args.key_file,
            collection_name=args.collection_name,
            allow_empty_collection=False,
        )

        res = rag.answer_question(args.question, top_k=args.top_k, temperature=0.1, max_tokens=192)
        answer = (res.get("answer") or "").strip()

        if _has_steps(answer):
            print("FAIL: output contains Step structure (forbidden)")
            print(answer)
            return 2

        if not _has_answer_explanation(answer):
            print("FAIL: output missing 'Answer:' and 'Explanation:' lines")
            print(answer)
            return 2

        expl = _extract_explanation_line(answer)
        if not _is_one_sentence_explanation(expl):
            print("FAIL: explanation is not concise (expected one short sentence)")
            print(expl)
            return 2

        if "i don't know" in answer.lower():
            print("FAIL: model answered I don't know (should only do so if not derivable)")
            print(answer)
            return 2

        final_block = _extract_final_answer_block(answer)
        if not _contains_datetime_like(final_block):
            print("FAIL: Answer line doesn't look like a datetime")
            print("Answer line:\n", final_block)
            return 2

        print("PASS")
        print(answer)
        return 0

    except Exception as e:
        print("RUNTIME ERROR:", repr(e))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
