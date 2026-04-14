"""Evaluate llama2 vs mistral answers against a gold-answer file.

This is a generalized version for larger runs (e.g., 60 questions).

Input formats:
- compare JSONL: each line has fields: question, llama2_answer, mistral_answer
- queries file: one question per line (optional; if missing we assume JSONL order)
- gold file: one gold answer per line, aligned by line number with queries/JSONL order

Scoring heuristics:
- yes/no normalization
- datetime normalization: 20260301_13:00 or 2026-03-01 13:00 => 20260301_13:00
- date normalization: YYYYMMDD
- substring match
- token overlap (>=threshold)
- penalize "I don't know" / hedging

Outputs:
- JSON with per-item correctness and reasons
- Markdown report (thesis-friendly)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


IDK_PATTERNS = (
    "i don't know",
    "i do not know",
    "not enough information",
    "no information",
    "not provided",
    "cannot determine",
    "impossible to determine",
)


def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def is_idk(ans: str) -> bool:
    a = norm(ans)
    return any(p in a for p in IDK_PATTERNS)


def extract_yes_no(s: str) -> Optional[str]:
    t = norm(s)
    if re.search(r"\byes\b", t):
        return "yes"
    if re.search(r"\bno\b", t):
        return "no"
    return None


def extract_yyyymmdd(s: str) -> Optional[str]:
    t = norm(s)
    m = re.search(r"\b(20\d{2})[-/]?(\d{2})[-/]?(\d{2})\b", t)
    if not m:
        return None
    return f"{m.group(1)}{m.group(2)}{m.group(3)}"


def extract_datetime_compact(s: str) -> Optional[str]:
    """Normalize to YYYYMMDD_HH:MM if possible."""
    t = norm(s)

    # Already compact: 20260301_13:00
    m = re.search(r"\b(20\d{2})(\d{2})(\d{2})[_\s-]?(\d{2}):(\d{2})\b", t)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}_{m.group(4)}:{m.group(5)}"

    # With separators: 2026-03-01 13:00
    m = re.search(r"\b(20\d{2})[-/](\d{2})[-/](\d{2})\s+(\d{1,2}):(\d{2})\b", t)
    if m:
        hh = int(m.group(4))
        return f"{m.group(1)}{m.group(2)}{m.group(3)}_{hh:02d}:{m.group(5)}"

    return None


def token_set(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", norm(s)) if len(t) >= 2}


@dataclass
class Match:
    correct: bool
    reason: str


def match_answer(gold: str, ans: str, *, overlap_threshold: float = 0.6) -> Match:
    g = norm(gold)
    a = norm(ans)

    if not a:
        return Match(False, "empty")

    if not g:
        # empty gold is not expected
        return Match(False, "empty_gold")

    if is_idk(a):
        return Match(False, "idk")

    # yes/no
    if g in ("yes", "no"):
        yn = extract_yes_no(a)
        return Match(yn == g, f"yes/no pred={yn}")

    # datetime
    gdt = extract_datetime_compact(g)
    if gdt:
        adt = extract_datetime_compact(a)
        return Match(adt == gdt, f"datetime pred={adt}")

    # date
    gdate = extract_yyyymmdd(g) if re.search(r"20\d{6}", g) or re.search(r"20\d{2}[-/]\d{2}[-/]\d{2}", g) else None
    if gdate:
        adate = extract_yyyymmdd(a)
        return Match(adate == gdate, f"date pred={adate}")

    # substring either way for short answers
    if len(g) <= 80 and g in a:
        return Match(True, "substring")

    # Token overlap
    gt = token_set(g)
    at = token_set(a)
    if not gt:
        return Match(False, "no_gold_tokens")
    overlap = len(gt & at) / max(1, len(gt))
    return Match(overlap >= overlap_threshold, f"token_overlap={overlap:.2f}")


def load_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare_jsonl", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_md", required=True)
    ap.add_argument("--overlap_threshold", type=float, default=0.6)
    args = ap.parse_args()

    compare_path = Path(args.compare_jsonl)
    gold_path = Path(args.gold)
    queries_path = Path(args.queries) if args.queries else None
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)

    # Ensure output dirs exist
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    print(f"[eval] compare_jsonl={compare_path}")
    print(f"[eval] gold={gold_path}")
    if queries_path:
        print(f"[eval] queries={queries_path}")
    print(f"[eval] output_json={out_json}")
    print(f"[eval] output_md={out_md}")

    gold = load_lines(gold_path)
    print(f"[eval] gold_lines={len(gold)}")

    rows_text = compare_path.read_text(encoding='utf-8')
    rows = [json.loads(line) for line in rows_text.splitlines() if line.strip()]
    print(f"[eval] rows={len(rows)}")

    if len(rows) != len(gold):
        raise SystemExit(f"Gold length ({len(gold)}) != jsonl rows ({len(rows)})")

    queries = None
    if queries_path:
        queries = load_lines(queries_path)
        print(f"[eval] queries_lines={len(queries)}")
        if len(queries) != len(rows):
            raise SystemExit(f"Queries length ({len(queries)}) != jsonl rows ({len(rows)})")

    per = []
    llama_ok = 0
    mistral_ok = 0
    llama_idk = 0
    mistral_idk = 0

    for i, row in enumerate(rows):
        q = (queries[i] if queries else row.get("question") or "").strip()
        g = gold[i]
        l = row.get("llama2_answer") or ""
        m = row.get("mistral_answer") or ""

        lmatch = match_answer(g, l, overlap_threshold=float(args.overlap_threshold))
        mmatch = match_answer(g, m, overlap_threshold=float(args.overlap_threshold))

        llama_ok += int(lmatch.correct)
        mistral_ok += int(mmatch.correct)
        llama_idk += int(is_idk(l))
        mistral_idk += int(is_idk(m))

        per.append({
            "idx": i + 1,
            "question": q,
            "gold": g,
            "llama2_answer": l,
            "mistral_answer": m,
            "llama2_correct": lmatch.correct,
            "mistral_correct": mmatch.correct,
            "llama2_reason": lmatch.reason,
            "mistral_reason": mmatch.reason,
        })

    total = len(rows)
    summary = {
        "total": total,
        "llama2_correct": llama_ok,
        "mistral_correct": mistral_ok,
        "llama2_acc": llama_ok / total,
        "mistral_acc": mistral_ok / total,
        "llama2_idk_rate": llama_idk / total,
        "mistral_idk_rate": mistral_idk / total,
    }

    winner = "tie"
    if mistral_ok > llama_ok:
        winner = "mistral"
    elif llama_ok > mistral_ok:
        winner = "llama2"

    # JSON output
    out_json.write_text(
        json.dumps({"summary": summary, "winner": winner, "items": per}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Markdown report
    md: List[str] = []
    md.append("# LLM 对比评测报告（基于 Gold Answer，Single 类型）\n")
    md.append(f"输入对比文件：`{compare_path.as_posix()}`  ")
    md.append(f"Gold 文件：`{gold_path.as_posix()}`  ")
    if queries_path:
        md.append(f"Queries 文件：`{queries_path.as_posix()}`  ")
    md.append("\n")

    md.append("## 1. 定量结果\n")
    md.append(f"- llama2：**{llama_ok}/{total}**（{summary['llama2_acc']:.2%}）")
    md.append(f"- mistral：**{mistral_ok}/{total}**（{summary['mistral_acc']:.2%}）")
    md.append(f"- `I don't know` 比例：llama2={summary['llama2_idk_rate']:.2%}，mistral={summary['mistral_idk_rate']:.2%}\n")
    md.append(f"**本轮赢家（按命中数）：`{winner}`**\n")

    md.append("## 2. 质性分析（论文可用）\n")
    md.append("由于样本量为 60（Single 类型），结论具有参考意义但不具有绝对性。建议结合以下维度解释：\n")
    md.append("- **指令遵循**：是否严格只基于 context；是否出现超出上下文的推断/编造。\n")
    md.append("- **回答保守性**：缺乏证据时是否倾向输出 ‘I don't know’（更安全但可能降低覆盖率）。\n")
    md.append("- **信息覆盖**：在实体/列表题中是否遗漏关键实体、或引入多余实体。\n")
    md.append("- **可读性与简洁性**：是否短而明确，便于工程落地与日志分析。\n")

    md.append("## 3. 逐题判定（摘要）\n")
    md.append("| # | Gold | llama2 ✓ | mistral ✓ | 备注 |")
    md.append("|---:|---|:---:|:---:|---|")
    for it in per:
        note = ""
        if it["llama2_correct"] != it["mistral_correct"]:
            note = "model_diff"
        md.append(
            f"| {it['idx']} | {str(it['gold']).replace('|','\\|')} | {'✓' if it['llama2_correct'] else '✗'} | {'✓' if it['mistral_correct'] else '✗'} | {note} |"
        )

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("[eval] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
