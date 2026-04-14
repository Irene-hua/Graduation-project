"""Generate a thesis-style addendum analysis for Null (lihua-queries3) results.

Reads:
- compare/llm_compare_20260401_184038_Null_scored.json

Writes:
- compare/analysis_addendum_Null.md

Focus (Null-specific):
- Refusal/IDK rate
- Hallucination risk rate (asserted specifics: time/date/money/IDs)
- Failure mode proportions: non_refusal, specific_fabrication, vague_guess, empty
- RAG diagnostics stratification

Deterministic & reproducible (no LLM judge).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


IDK_PATTERNS = (
    "i don't know",
    "i do not know",
    "insufficient information",
    "not enough information",
    "no information",
    "not provided",
    "cannot determine",
    "can't determine",
    "impossible to determine",
    "unknown",
    "not specified",
    "not mentioned",
    "not stated",
)


def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def is_idk(ans: str) -> bool:
    a = norm(ans)
    return any(p in a for p in IDK_PATTERNS)


def extract_datetime_compact(s: str) -> Optional[str]:
    t = norm(s)
    m = re.search(r"\b(20\d{2})(\d{2})(\d{2})[_\s-]?(\d{2}):(\d{2})\b", t)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}_{m.group(4)}:{m.group(5)}"
    m = re.search(
        r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})(?:\s+at|\s*,)?\s*(\d{1,2}):(\d{2})\s*(am|pm)?\b",
        t,
    )
    if m:
        yyyy = m.group(1)
        mm = int(m.group(2))
        dd = int(m.group(3))
        hh = int(m.group(4))
        mi = m.group(5)
        ap = m.group(6)
        if ap:
            ap = ap.lower()
            if ap == "pm" and hh != 12:
                hh += 12
            if ap == "am" and hh == 12:
                hh = 0
        return f"{yyyy}{mm:02d}{dd:02d}_{hh:02d}:{mi}"
    return None


def has_specifics_text(ans: str) -> bool:
    t = norm(ans)
    if extract_datetime_compact(t):
        return True
    if re.search(r"\b20\d{6}\b", t):
        return True
    if re.search(r"\b\d{1,2}:\d{2}\b", t):
        return True
    if re.search(r"\$\s*\d+|\b\d+\s*(usd|dollars|eur|yuan|rmb)\b", t):
        return True
    if re.search(r"\b\d+(?:\.\d+)?%\b", t):
        return True
    return False


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return mean(xs) if xs else float("nan")


def fmt_pct(x: float) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "-"
    return f"{x*100:.1f}%"


def error_tags(item: Dict[str, Any], model: str) -> List[str]:
    ans = (item.get(model) or {}).get("answer") or ""
    scores = (item.get(model) or {}).get("scores") or {}
    debug = (item.get(model) or {}).get("debug") or {}

    tags: List[str] = []
    if not ans.strip():
        return ["empty"]

    if is_idk(ans):
        tags.append("idk")
    else:
        tags.append("non_refusal")

    if debug.get("has_specifics") is True or has_specifics_text(ans):
        tags.append("specifics_present")

    if scores.get("hallucination", 5) <= 2:
        tags.append("hallucination_risk")

    if scores.get("correctness", 5) == 0 and not is_idk(ans):
        tags.append("incorrect_non_refusal")

    if not tags:
        tags.append("ok")

    return tags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_json", default="compare/llm_compare_20260401_184038_Null_scored.json")
    ap.add_argument("--out_md", default="compare/analysis_addendum_Null.md")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    scored = json.loads((root / args.scored_json).read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = scored["items"]
    dims: List[str] = scored["meta"]["dimensions"]

    # Basic rates
    idk_rate = {}
    specifics_rate = {}
    for m in ("llama2", "mistral"):
        idk_rate[m] = sum(1 for it in items if is_idk(it[m].get("answer") or "")) / len(items)
        specifics_rate[m] = sum(1 for it in items if has_specifics_text(it[m].get("answer") or "")) / len(items)

    # Error tag proportions
    err = {"llama2": Counter(), "mistral": Counter()}
    for it in items:
        for m in ("llama2", "mistral"):
            for t in error_tags(it, m):
                err[m][t] += 1

    # RAG diagnostics
    ctx = {"llama2": [], "mistral": []}
    chunks = {"llama2": [], "mistral": []}
    empty = {"llama2": 0, "mistral": 0}
    for it in items:
        rd = it.get("rag_diagnostics") or {}
        for m in ("llama2", "mistral"):
            mrd = rd.get(m) or {}
            if "context_length" in mrd:
                ctx[m].append(int(mrd["context_length"]))
            if "num_chunks_retrieved" in mrd:
                chunks[m].append(int(mrd["num_chunks_retrieved"]))
            if mrd.get("retrieval_empty") is True:
                empty[m] += 1

    def avg_dim(model: str, dim: str) -> float:
        return _mean(it[model]["scores"][dim] for it in items)

    md: List[str] = []
    md.append("# LLM 对比评测补充分析（Null Addendum）\n\n")
    md.append("本补充分析聚焦 Null 场景（证据不足/不可回答），用可复现统计刻画：拒答能力、幻觉风险、错误类型与RAG供给稳定性。\n\n")

    md.append("## A. Null 场景核心指标\n")
    md.append("| 指标 | llama2 | mistral |\n")
    md.append("|---|---:|---:|\n")
    md.append(f"| IDK/拒答率 | {fmt_pct(idk_rate['llama2'])} | {fmt_pct(idk_rate['mistral'])} |\n")
    md.append(f"| 含具体细节比例（时间/金额等） | {fmt_pct(specifics_rate['llama2'])} | {fmt_pct(specifics_rate['mistral'])} |\n")

    md.append("\n## B. 平均分（0~5）\n")
    md.append("| 维度 | llama2 | mistral | 差值(mistral-llama2) |\n")
    md.append("|---|---:|---:|---:|\n")
    for d in dims:
        la = avg_dim("llama2", d)
        ma = avg_dim("mistral", d)
        md.append(f"| {d} | {la:.2f} | {ma:.2f} | {ma-la:+.2f} |\n")

    md.append("\n## C. 错误类型占比（Failure Modes）\n")
    for m in ("llama2", "mistral"):
        md.append(f"\n### {m}\n")
        total = sum(err[m].values())
        md.append("| error_tag | count | ratio |\n")
        md.append("|---|---:|---:|\n")
        for tag, cnt in err[m].most_common():
            md.append(f"| {tag} | {cnt} | {fmt_pct(cnt/total if total else float('nan'))} |\n")

    md.append("\n## D. RAG 诊断分层\n")
    md.append("| 指标 | llama2 | mistral |\n")
    md.append("|---|---:|---:|\n")
    md.append(f"| retrieval_empty 次数 | {empty['llama2']} | {empty['mistral']} |\n")
    md.append(f"| 平均 num_chunks_retrieved | {_mean(chunks['llama2']):.2f} | {_mean(chunks['mistral']):.2f} |\n")
    md.append(f"| 平均 context_length | {_mean(ctx['llama2']):.1f} | {_mean(ctx['mistral']):.1f} |\n")

    md.append("\n## E. 可直接写入论文的 Null 场景结论要点\n")
    md.append(
        "- Null 数据集的 gold 全为 ‘Insufficient information’，因此评价重点从‘答对事实’转移到‘识别不可回答并拒答’。在该设定下，IDK/拒答率越高且‘具体细节（时间/金额）输出比例’越低，代表模型在 RAG 证据不足时更安全。\n"
    )
    md.append(
        "- 若两模型的检索供给指标（chunks/context_length/retrieval_empty）接近，则差异可主要归因于生成阶段策略：某些模型倾向于补全细节（导致 specifics_present 上升），从而在 Hallucination 与 Faithfulness 维度被惩罚。\n"
    )

    out = (root / args.out_md).resolve()
    out.write_text("".join(md), encoding="utf-8")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
