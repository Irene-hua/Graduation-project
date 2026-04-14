"""Evaluate llama2 vs mistral answers against a small gold set.

Usage (PowerShell):
  python scripts\evaluate_llm_compare_with_gold.py \
    --input results\llm_compare_20260331_184522.jsonl \
    --output results\llm_compare_20260331_184522_scored.json \
    --report docs\LLM_Comparison_Report.md

Gold answers are embedded for the 20-question LiHua-World evaluation described in the thesis.

Scoring is intentionally lightweight:
- binary-ish match with tolerance (substring / token overlap / date normalization / yes-no)
- plus safety checks ("I don't know" / obvious hedging)

This script does NOT change the RAG pipeline. It's purely offline evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


GOLD: List[str] = [
    "yes",
    "no",
    "yes",
    "no",
    "20260122",
    "yes",
    "wolfgang",
    "lihua & chae & yuriko",
    "adam smith & jennifer moore & wolfgang schulz",
    "about 27 days",
    "haileyjohnson",
    "lihuawei introduce them to each other by saying that they can play music together every sunday",
    "a fresh sourdough loaf and a bottle of milk and lihua praises hailey's bread and milk",
    "yes",
    "yes",
    "keep noise to a minimum during late hours and take good care of the property",
    "the cozy café downtown",
    "20260108",
    "fitzone",
    "20260118",
]


YES = {"yes", "y", "true"}
NO = {"no", "n", "false"}


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_yes_no(ans: str) -> str | None:
    a = _norm(ans)
    # Accept common forms.
    if re.search(r"\byes\b", a):
        return "yes"
    if re.search(r"\bno\b", a):
        return "no"
    return None


def _extract_yyyymmdd(ans: str) -> str | None:
    a = _norm(ans)
    # Matches 20260122 or 2026-01-22 or 2026/01/22
    m = re.search(r"\b(20\d{2})[-/]?(\d{2})[-/]?(\d{2})\b", a)
    if not m:
        return None
    return f"{m.group(1)}{m.group(2)}{m.group(3)}"


def _is_idk(ans: str) -> bool:
    a = _norm(ans)
    return ("i don't know" in a) or ("i do not know" in a) or ("not enough information" in a) or ("no information" in a)


def _token_set(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", _norm(s)) if len(t) >= 2}


@dataclass
class MatchResult:
    correct: bool
    reason: str


def match(gold: str, ans: str) -> MatchResult:
    g = _norm(gold)
    a = _norm(ans)

    if not a:
        return MatchResult(False, "empty")

    # If model says IDK, almost always incorrect unless gold is also unknown (not the case here)
    if _is_idk(a):
        return MatchResult(False, "idk")

    # Yes/No questions
    if g in ("yes", "no"):
        yn = _extract_yes_no(a)
        if yn == g:
            return MatchResult(True, "yes/no match")
        return MatchResult(False, f"yes/no mismatch (pred={yn})")

    # Date questions
    g_date = _extract_yyyymmdd(g) or (g if re.fullmatch(r"\d{8}", g) else None)
    if g_date:
        a_date = _extract_yyyymmdd(a)
        if a_date == g_date:
            return MatchResult(True, "date match")
        # tolerate month-day mention without year? (not needed here)
        return MatchResult(False, f"date mismatch (pred={a_date})")

    # Simple containment
    if g and (g in a):
        return MatchResult(True, "substring")

    # Token overlap (for entity lists / paraphrases)
    g_tok = _token_set(g)
    a_tok = _token_set(a)
    if not g_tok:
        return MatchResult(False, "no gold tokens")
    overlap = len(g_tok & a_tok) / max(1, len(g_tok))
    if overlap >= 0.6:
        return MatchResult(True, f"token overlap {overlap:.2f}")

    return MatchResult(False, f"token overlap {overlap:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    in_path = Path(args.input)
    rows = [json.loads(line) for line in in_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if len(rows) != 20:
        raise SystemExit(f"Expected 20 rows, got {len(rows)}")

    scored_rows: List[Dict] = []
    llama_correct = 0
    mistral_correct = 0

    for i, row in enumerate(rows):
        gold = GOLD[i]
        l = row.get("llama2_answer", "")
        m = row.get("mistral_answer", "")

        l_res = match(gold, l)
        m_res = match(gold, m)

        llama_correct += int(l_res.correct)
        mistral_correct += int(m_res.correct)

        scored_rows.append({
            **row,
            "gold": gold,
            "llama2_correct": l_res.correct,
            "llama2_match_reason": l_res.reason,
            "mistral_correct": m_res.correct,
            "mistral_match_reason": m_res.reason,
        })

    summary = {
        "file": str(in_path).replace("\\", "/"),
        "total": len(rows),
        "llama2_correct": llama_correct,
        "mistral_correct": mistral_correct,
        "llama2_acc": llama_correct / len(rows),
        "mistral_acc": mistral_correct / len(rows),
    }

    Path(args.output).write_text(json.dumps({"summary": summary, "rows": scored_rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown report
    lines: List[str] = []
    lines.append("# LLM Comparison Report (Gold-Answer Evaluation)\n")
    lines.append(f"Input: `{args.input}`  ")
    lines.append(f"Date: 2026-03-31\n")
    lines.append("## Summary\n")
    lines.append(f"- llama2 correct: **{llama_correct}/{len(rows)}** (acc={summary['llama2_acc']:.2%})")
    lines.append(f"- mistral correct: **{mistral_correct}/{len(rows)}** (acc={summary['mistral_acc']:.2%})\n")

    winner = "tie"
    if mistral_correct > llama_correct:
        winner = "mistral"
    elif llama_correct > mistral_correct:
        winner = "llama2"

    lines.append(f"**Winner on this 20-question set:** `{winner}`\n")

    lines.append("## Per-question judgments\n")
    lines.append("| # | Question (short) | Gold | llama2 | llama2 ok | mistral | mistral ok |")
    lines.append("|---:|---|---|---|:---:|---|:---:|")
    for i, r in enumerate(scored_rows, start=1):
        q = (r.get("question") or "").replace("|", "\\|")
        qshort = (q[:80] + "…") if len(q) > 80 else q
        gold = (r.get("gold") or "").replace("|", "\\|")
        l = (str(r.get("llama2_answer") or "").strip().replace("\n", " "))
        m = (str(r.get("mistral_answer") or "").strip().replace("\n", " "))
        lshort = (l[:60] + "…") if len(l) > 60 else l
        mshort = (m[:60] + "…") if len(m) > 60 else m
        lines.append(
            f"| {i} | {qshort} | {gold} | {lshort.replace('|','\\|')} | {'✓' if r['llama2_correct'] else '✗'} | {mshort.replace('|','\\|')} | {'✓' if r['mistral_correct'] else '✗'} |"
        )

    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

