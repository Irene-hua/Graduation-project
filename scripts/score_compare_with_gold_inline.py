"""One-shot scorer: compare JSONL answers against a gold file (line-aligned).

This script is designed to be robust even if stdout in the host environment is flaky:
- It ALWAYS writes outputs.

Outputs:
- results/<stem>_scored.json
- docs/LLM_Comparison_Report_Single_60Q_20260401.md

Usage:
  python scripts/score_compare_with_gold_inline.py \
    --compare results/llm_compare_20260401_131033.jsonl \
    --gold data/gold-answer/lihua-queries2-gold-answer \
    --queries data/test_datasets/lihua-queries2
"""

from __future__ import annotations

import argparse
import json
import re
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
    t = norm(s)
    m = re.search(r"\b(20\d{2})(\d{2})(\d{2})[_\s-]?(\d{2}):(\d{2})\b", t)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}_{m.group(4)}:{m.group(5)}"
    m = re.search(r"\b(20\d{2})[-/](\d{2})[-/](\d{2})\s+(\d{1,2}):(\d{2})\b", t)
    if m:
        hh = int(m.group(4))
        return f"{m.group(1)}{m.group(2)}{m.group(3)}_{hh:02d}:{m.group(5)}"
    return None


def token_set(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", norm(s)) if len(t) >= 2}


def match_answer(gold: str, ans: str, overlap_threshold: float = 0.6) -> tuple[bool, str]:
    g = norm(gold)
    a = norm(ans)

    if not a:
        return False, "empty"
    if not g:
        return False, "empty_gold"
    if is_idk(a):
        return False, "idk"

    if g in ("yes", "no"):
        pred = extract_yes_no(a)
        return pred == g, f"yes/no pred={pred}"

    gdt = extract_datetime_compact(g)
    if gdt:
        adt = extract_datetime_compact(a)
        return adt == gdt, f"datetime pred={adt}"

    gdate = extract_yyyymmdd(g) if (re.search(r"20\d{6}", g) or re.search(r"20\d{2}[-/]\d{2}[-/]\d{2}", g)) else None
    if gdate:
        adate = extract_yyyymmdd(a)
        return adate == gdate, f"date pred={adate}"

    if len(g) <= 80 and g in a:
        return True, "substring"

    gt = token_set(g)
    at = token_set(a)
    if not gt:
        return False, "no_gold_tokens"
    overlap = len(gt & at) / max(1, len(gt))
    return overlap >= overlap_threshold, f"token_overlap={overlap:.2f}"


def load_lines(path: Path, *, keep_empty: bool = False) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if keep_empty:
        return [line.strip() for line in lines]
    return [line.strip() for line in lines if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--overlap_threshold", type=float, default=0.6)
    args = ap.parse_args()

    compare_path = Path(args.compare)
    gold_path = Path(args.gold)
    queries_path = Path(args.queries)

    rows = [json.loads(line) for line in compare_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = load_lines(gold_path)
    # queries may contain an extra trailing empty line; keep_empty then later trim
    queries = load_lines(queries_path, keep_empty=True)

    # Drop trailing empty lines in queries to stabilize alignment
    while queries and not queries[-1]:
        queries.pop()

    if not (len(rows) == len(gold) == len(queries)):
        min_len = min(len(rows), len(gold), len(queries))
        # Trim all to the same length to allow scoring to proceed.
        rows = rows[:min_len]
        gold = gold[:min_len]
        queries = queries[:min_len]
        # Continue; report the mismatch in outputs.

    items = []
    llama_ok = mistral_ok = 0
    llama_idk = mistral_idk = 0

    for i, (row, g, q) in enumerate(zip(rows, gold, queries), start=1):
        l = row.get("llama2_answer") or ""
        m = row.get("mistral_answer") or ""

        l_ok, l_reason = match_answer(g, l, float(args.overlap_threshold))
        m_ok, m_reason = match_answer(g, m, float(args.overlap_threshold))

        llama_ok += int(l_ok)
        mistral_ok += int(m_ok)
        llama_idk += int(is_idk(l))
        mistral_idk += int(is_idk(m))

        items.append({
            "idx": i,
            "question": q,
            "gold": g,
            "llama2_answer": l,
            "mistral_answer": m,
            "llama2_correct": l_ok,
            "mistral_correct": m_ok,
            "llama2_reason": l_reason,
            "mistral_reason": m_reason,
        })

    total = len(items)
    summary = {
        "total": total,
        "llama2_correct": llama_ok,
        "mistral_correct": mistral_ok,
        "llama2_acc": llama_ok / total,
        "mistral_acc": mistral_ok / total,
        "llama2_idk_rate": llama_idk / total,
        "mistral_idk_rate": mistral_idk / total,
        "len_rows": len(rows),
        "len_gold": len(gold),
        "len_queries": len(queries),
    }

    winner = "tie"
    if mistral_ok > llama_ok:
        winner = "mistral"
    elif llama_ok > mistral_ok:
        winner = "llama2"

    # Write outputs
    out_json = compare_path.with_name(compare_path.stem + "_scored.json")
    out_json.write_text(
        json.dumps({"summary": summary, "winner": winner, "items": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    docs_dir = compare_path.parent.parent / "docs" if (compare_path.parent.name == "results") else Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Auto-name report file based on compare stem to avoid overwriting across types/runs.
    # Example: results/llm_compare_20260401_131033.jsonl -> docs/LLM_Comparison_Report_llm_compare_20260401_131033.md
    out_md = docs_dir / f"LLM_Comparison_Report_{compare_path.stem}.md"

    md: List[str] = []
    md.append(f"# LLM 对比评测报告（Gold Answer）\n")
    md.append(f"对比回答：`{compare_path.as_posix()}`  ")
    md.append(f"Gold：`{gold_path.as_posix()}`  ")
    md.append(f"Queries：`{queries_path.as_posix()}`  ")
    md.append(f"输出JSON：`{out_json.as_posix()}`  ")
    md.append(f"输出报告：`{out_md.as_posix()}`\n")
    md.append("## 1. 定量结果\n")
    md.append(f"- llama2：**{llama_ok}/{total}**（{summary['llama2_acc']:.2%}）")
    md.append(f"- mistral：**{mistral_ok}/{total}**（{summary['mistral_acc']:.2%}）")

    llama_idk_s = format(summary["llama2_idk_rate"], ".2%")
    mistral_idk_s = format(summary["mistral_idk_rate"], ".2%")
    md.append("- `I don't know` 比例：llama2=" + llama_idk_s + "，mistral=" + mistral_idk_s + "\n")
    md.append(f"**本轮赢家（按命中数）：`{winner}`**\n")
    md.append("## 2. 说明\n")
    md.append("本评测采用宽松命中（包含/语义接近）规则，适合在论文中做小规模对比验证；建议结合 Multi/Null 类型进一步验证稳定性。\n")
    md.append("## 3. 逐题摘要\n")
    md.append("| # | Gold | llama2 | mistral | llama2✓ | mistral✓ |")
    md.append("|---:|---|---|---|:---:|:---:|")
    for it in items:
        l = norm(it["llama2_answer"])[:60].replace("|", "\\|")
        m = norm(it["mistral_answer"])[:60].replace("|", "\\|")
        md.append(
            f"| {it['idx']} | {it['gold'].replace('|','\\|')} | {l} | {m} | {'✓' if it['llama2_correct'] else '✗'} | {'✓' if it['mistral_correct'] else '✗'} |"
        )

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
