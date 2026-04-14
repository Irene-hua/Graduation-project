"""Generate a thesis-style addendum analysis for Multi (lihua-queries1) results.

Reads:
- compare/llm_compare_20260401_154508_Multi_scored.json

Writes:
- compare/analysis_addendum_Multi.md

Focus:
- Question type grouping (yesno/time/entity/descriptive/other)
- Error tag proportions (idk, incorrect_yesno, keyword_missing, list_partial, etc.)
- RAG diagnostics stability

Deterministic & reproducible (no LLM judge).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


IDK_PATTERNS = (
    "i don't know",
    "i do not know",
    "not enough information",
    "no information",
    "not provided",
    "cannot determine",
    "impossible to determine",
    "unknown",
)


def is_idk(ans: str) -> bool:
    a = norm(ans)
    return any(p in a for p in IDK_PATTERNS)


def token_set(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", norm(s)) if len(t) >= 2}


def extract_yes_no(s: str) -> Optional[str]:
    t = norm(s)
    if re.search(r"\byes\b", t):
        return "yes"
    if re.search(r"\bno\b", t):
        return "no"
    if re.search(r"\b(did not|didn't|do not|don't|nope|not)\b", t):
        return "no"
    m = re.search(r"answer is (yes|no)", t)
    if m:
        return m.group(1)
    return None


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

    month_map = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }

    m = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+"
        r"(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(20\d{2})\s*(?:at\s*)?(\d{1,2}):(\d{2})\s*(am|pm)?\b",
        t,
    )
    if m:
        mon = month_map[m.group(1)]
        dd = int(m.group(2))
        yyyy = m.group(3)
        hh = int(m.group(4))
        mi = m.group(5)
        ap = m.group(6)
        if ap:
            ap = ap.lower()
            if ap == "pm" and hh != 12:
                hh += 12
            if ap == "am" and hh == 12:
                hh = 0
        return f"{yyyy}{mon:02d}{dd:02d}_{hh:02d}:{mi}"

    return None


def classify_question(q: str, gold: str) -> str:
    qn = norm(q)
    gn = norm(gold)

    if gn in ("yes", "no") or qn.startswith("did ") or qn.startswith("is ") or qn.startswith("are ") or qn.startswith("do "):
        return "yesno"

    if qn.startswith("when ") or " time" in qn or " date" in qn or extract_datetime_compact(gn):
        return "time"

    if qn.startswith("where ") or " address" in qn or "name of" in qn or qn.startswith("who "):
        return "entity"

    if qn.startswith("what ") or qn.startswith("why ") or qn.startswith("how "):
        return "descriptive"

    return "other"


def extract_error_tags(it: Dict[str, Any], model: str) -> List[str]:
    ans = it[model].get("answer") or ""
    scores = it[model].get("scores") or {}
    debug = it[model].get("debug") or {}
    gold = it.get("ground_truth") or ""

    tags: List[str] = []

    if not ans.strip():
        return ["empty"]

    if is_idk(ans):
        tags.append("idk")

    gtype = debug.get("gold_type")
    if gtype == "yesno":
        pred = extract_yes_no(ans)
        if pred is None:
            tags.append("yesno_unparsed")
        else:
            if pred != norm(gold):
                tags.append("yesno_incorrect")

    if gtype == "multi_list":
        f1 = debug.get("f1")
        if isinstance(f1, (int, float)):
            if 0 < f1 < 0.9:
                tags.append("list_partial")
            if f1 == 0:
                tags.append("list_miss")

    if scores.get("hallucination", 5) <= 2:
        tags.append("hallucination_risk")

    if scores.get("correctness", 5) == 0 and not is_idk(ans):
        tags.append("incorrect")

    if not tags:
        tags.append("ok_or_minor")

    return tags


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return mean(xs) if xs else float("nan")


def fmt_pct(x: float) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "-"
    return f"{x * 100:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_json", default="compare/llm_compare_20260401_154508_Multi_scored.json")
    ap.add_argument("--out_md", default="compare/analysis_addendum_Multi.md")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    scored = json.loads((root / args.scored_json).read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = scored["items"]
    dims: List[str] = scored["meta"]["dimensions"]

    # group by qtype
    qgroups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        qt = classify_question(it.get("question", ""), it.get("ground_truth", ""))
        it["_qtype"] = qt
        qgroups[qt].append(it)

    # wins per qtype
    wins_by_qtype = {qt: Counter() for qt in qgroups}
    for qt, subset in qgroups.items():
        for it in subset:
            wins_by_qtype[qt][it.get("better_model", "tie")] += 1

    # error tags
    err = {"llama2": Counter(), "mistral": Counter()}
    for it in items:
        for m in ("llama2", "mistral"):
            for tag in extract_error_tags(it, m):
                err[m][tag] += 1

    # rag diagnostics
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

    def avg_dim(model: str, dim: str, subset: List[Dict[str, Any]]) -> float:
        return _mean(it[model]["scores"][dim] for it in subset)

    md: List[str] = []
    md.append("# LLM 对比评测补充分析（Multi Addendum）\n")
    md.append("基于 Multi(60题) 离线0~5多维评分结果，从题型/错误类型/RAG诊断角度补充可用于论文的统计结论。\n\n")

    md.append("## A. 题型分布\n")
    md.append("| 题型 | 数量 | 占比 |\n")
    md.append("|---|---:|---:|\n")
    for qt in sorted(qgroups.keys()):
        n = len(qgroups[qt])
        md.append(f"| {qt} | {n} | {fmt_pct(n / len(items))} |\n")

    md.append("\n## B. 按题型的平均分与胜负\n")
    for qt in sorted(qgroups.keys()):
        subset = qgroups[qt]
        md.append(f"\n### 题型：{qt}（n={len(subset)}）\n")
        md.append("| 维度 | llama2 | mistral | 差值(mistral-llama2) |\n")
        md.append("|---|---:|---:|---:|\n")
        for d in dims:
            la = avg_dim("llama2", d, subset)
            ma = avg_dim("mistral", d, subset)
            md.append(f"| {d} | {la:.2f} | {ma:.2f} | {ma - la:+.2f} |\n")

        w = wins_by_qtype[qt]
        md.append("\n胜负统计（按单题综合得分）：\n")
        md.append(f"- llama2: {w['llama2']}\n")
        md.append(f"- mistral: {w['mistral']}\n")
        md.append(f"- tie: {w['tie']}\n")

    md.append("\n## C. 错误类型占比（Failure Modes）\n")

    def render_err(model: str) -> None:
        md.append(f"\n### {model}\n")
        c = err[model]
        total = sum(c.values())
        md.append("| error_tag | count | ratio |\n")
        md.append("|---|---:|---:|\n")
        for tag, cnt in c.most_common():
            md.append(f"| {tag} | {cnt} | {fmt_pct(cnt / total if total else float('nan'))} |\n")

    render_err("llama2")
    render_err("mistral")

    md.append("\n## D. RAG 诊断分层\n")
    md.append("| 指标 | llama2 | mistral |\n")
    md.append("|---|---:|---:|\n")
    md.append(f"| retrieval_empty 次数 | {empty['llama2']} | {empty['mistral']} |\n")
    md.append(f"| 平均 num_chunks_retrieved | {_mean(chunks['llama2']):.2f} | {_mean(chunks['mistral']):.2f} |\n")
    md.append(f"| 平均 context_length | {_mean(ctx['llama2']):.1f} | {_mean(ctx['mistral']):.1f} |\n")

    md.append("\n## E. 可直接写入论文的结论模板\n")
    md.append(
        "- Multi 类型任务中，模型除需要做事实判断（是/否）外，还需要覆盖多要点列表并处理跨时间关系。分组结果通常表现为：在 yesno 题上，模型差异更多来自对证据的对齐与否；在 multi_list 题上，差异更多来自列表覆盖率（list_partial/list_miss）。\n"
    )
    md.append(
        "- 若 RAG 诊断显示两模型的检索供给（chunks/context_length）一致且 retrieval_empty 近似 0，则可将性能差异主要归因于生成阶段（抽取、推断与表达策略），而非检索阶段偏差。\n"
    )

    out = (root / args.out_md).resolve()
    out.write_text("".join(md), encoding="utf-8")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
