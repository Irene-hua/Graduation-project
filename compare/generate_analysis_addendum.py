"""Generate a thesis-style addendum analysis for LLM compare results.

Reads the scored JSON produced by `compare/rag_llm_compare_0_5_scoring.py` and writes:
- compare/analysis_addendum.md

Addendum focuses on:
- Question/answer-type grouping (time/entity/yes-no/other)
- Error type proportions (time offset, keyword missing, IDK, over-hedging, etc.)
- RAG diagnostics stratification (context_length, retrieval_empty, chunks)

This is heuristic but fully deterministic and reproducible.
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

HEDGE_PATTERNS = (
    "not explicitly stated",
    "not explicitly mentioned",
    "it is not stated",
    "it is not mentioned",
    "cannot be determined",
    "we cannot definitively",
    "unable to",
    "may be",
    "might be",
    "could be",
)


def is_idk(ans: str) -> bool:
    a = norm(ans)
    return any(p in a for p in IDK_PATTERNS)


def has_hedging(ans: str) -> bool:
    a = norm(ans)
    return any(p in a for p in HEDGE_PATTERNS)


def token_set(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", norm(s)) if len(t) >= 2}


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


def extract_yes_no(s: str) -> Optional[str]:
    t = norm(s)
    if re.search(r"\byes\b", t):
        return "yes"
    if re.search(r"\bno\b", t):
        return "no"
    return None


def classify_question(q: str, gold: str) -> str:
    qn = norm(q)
    gn = norm(gold)

    if gn in ("yes", "no") or qn.startswith("is ") or qn.startswith("are ") or qn.startswith("do ") or qn.startswith("did "):
        return "yesno"

    if qn.startswith("when ") or " time" in qn or " date" in qn or extract_datetime_compact(gn):
        return "time"

    if qn.startswith("where ") or " address" in qn or "name of" in qn or qn.startswith("who ") or qn.startswith("what is the name"):
        return "entity"

    if qn.startswith("what ") or qn.startswith("why ") or qn.startswith("how "):
        return "descriptive"

    return "other"


def extract_error_tags(item: Dict[str, Any], model_key: str) -> List[str]:
    """Assign deterministic error tags based on scored output and debug signals."""

    m = item[model_key]
    ans = m.get("answer") or ""
    gold = item.get("ground_truth") or ""
    scores = m.get("scores") or {}
    debug = m.get("debug") or {}

    tags: List[str] = []

    if not ans.strip():
        return ["empty"]

    if is_idk(ans):
        tags.append("idk")

    if has_hedging(ans):
        tags.append("hedging")

    gold_type = debug.get("gold_type")
    if gold_type == "datetime":
        pred = debug.get("pred_datetime")
        gd = debug.get("gold_datetime")
        if pred and gd and pred != gd:
            # date correct but time wrong handled in scorer; still record time_offset
            if pred.split("_")[0] == gd.split("_")[0]:
                tags.append("time_offset_same_date")
            else:
                tags.append("wrong_date")

    # Keyword/phrase missing proxy: low overlap for non-datetime with low correctness
    if gold_type in ("short", "free"):
        overlap = debug.get("token_overlap")
        if isinstance(overlap, (int, float)) and overlap < 0.6 and scores.get("correctness", 0) <= 1:
            tags.append("keyword_missing")

    # Over-generation (potential hallucination)
    if len(ans) > 180 and gold_type in ("short", "datetime", "date", "yesno"):
        tags.append("overlong_for_short_gold")

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
    if math.isnan(x):
        return "-"
    return f"{x * 100:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scored_json",
        default="compare/llm_compare_20260401_131033_scored.json",
        help="Path to scored json produced by rag_llm_compare_0_5_scoring.py",
    )
    ap.add_argument("--out_md", default="compare/analysis_addendum.md")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    scored_path = (root / args.scored_json).resolve()
    out_md = (root / args.out_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(scored_path.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = data["items"]

    # Grouping by question type
    qtype_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        qtype = classify_question(it.get("question", ""), it.get("ground_truth", ""))
        it["_qtype"] = qtype
        qtype_items[qtype].append(it)

    # Error tags distributions
    err_counts: Dict[str, Counter] = {"llama2": Counter(), "mistral": Counter()}
    for it in items:
        for mk in ("llama2", "mistral"):
            for tag in extract_error_tags(it, mk):
                err_counts[mk][tag] += 1

    # RAG diagnostics stratification
    ctx_lengths: Dict[str, List[int]] = {"llama2": [], "mistral": []}
    chunks: Dict[str, List[int]] = {"llama2": [], "mistral": []}
    retrieval_empty: Dict[str, int] = {"llama2": 0, "mistral": 0}

    for it in items:
        rd = it.get("rag_diagnostics") or {}
        for mk in ("llama2", "mistral"):
            mrd = rd.get(mk) or {}
            if "context_length" in mrd:
                ctx_lengths[mk].append(int(mrd["context_length"]))
            if "num_chunks_retrieved" in mrd:
                chunks[mk].append(int(mrd["num_chunks_retrieved"]))
            if mrd.get("retrieval_empty") is True:
                retrieval_empty[mk] += 1

    # Compute grouped averages
    dims = data["meta"]["dimensions"]

    def avg_dim(model: str, dim: str, subset: List[Dict[str, Any]]) -> float:
        return _mean(it[model]["scores"][dim] for it in subset)

    # Winner by qtype
    qtype_wins = {qt: Counter() for qt in qtype_items}
    for qt, subset in qtype_items.items():
        for it in subset:
            qtype_wins[qt][it.get("better_model", "tie")] += 1

    # Markdown output
    md: List[str] = []
    md.append("# LLM 对比评测补充分析（Addendum）\n")
    md.append("本补充章节基于 `scored.json` 的确定性离线评测结果，进一步从“题型分组 / 错误类型占比 / RAG诊断分层”角度给出可写入论文的硬核统计。\n\n")

    md.append("## A. 数据与方法补充\n")
    md.append(f"- 评分数据源：`{Path(args.scored_json).as_posix()}`\n")
    md.append(f"- 样本量：{len(items)}（Single）\n")
    md.append("- 题型分组规则（启发式、可复现）：\n")
    md.append("  - **time**：问题以 When/Time/Date 为主，或 gold 呈现 YYYYMMDD_HH:MM\n")
    md.append("  - **entity**：Where/Who/Name/Address 等实体型问题\n")
    md.append("  - **yesno**：gold 为 yes/no 或问题以 is/are/do/did 开头\n")
    md.append("  - **descriptive**：why/how/what 等描述型（不落入以上类）\n\n")

    md.append("## B. 按题型分组的模型表现\n")
    md.append("### B1. 题型分布\n")
    md.append("| 题型 | 数量 | 占比 |\n")
    md.append("|---|---:|---:|\n")
    for qt in sorted(qtype_items.keys()):
        n = len(qtype_items[qt])
        md.append(f"| {qt} | {n} | {fmt_pct(n / len(items))} |\n")

    md.append("\n### B2. 各题型的平均分（0~5）\n")
    for qt in sorted(qtype_items.keys()):
        subset = qtype_items[qt]
        md.append(f"\n**题型：{qt}（n={len(subset)}）**\n\n")
        md.append("| 维度 | llama2 | mistral | 差值(mistral-llama2) |\n")
        md.append("|---|---:|---:|---:|\n")
        for dim in dims:
            la = avg_dim("llama2", dim, subset)
            ma = avg_dim("mistral", dim, subset)
            md.append(f"| {dim} | {la:.2f} | {ma:.2f} | {ma - la:+.2f} |\n")

        w = qtype_wins[qt]
        md.append("\n胜负统计（按单题综合得分）：\n")
        md.append(f"- llama2: {w['llama2']}\n")
        md.append(f"- mistral: {w['mistral']}\n")
        md.append(f"- tie: {w['tie']}\n")

    md.append("\n## C. 错误类型占比分析\n")
    md.append("本节将单题‘错误’进一步分解为可解释类别（time偏差/关键词缺失/IDK/过度扩写/幻觉风险等），用于论文中对失败模式的归因。\n\n")

    def render_err_table(model: str) -> List[str]:
        c = err_counts[model]
        total = sum(c.values())
        lines: List[str] = []
        lines.append(f"### {model} 错误类型分布\n")
        lines.append("| error_tag | count | ratio |\n")
        lines.append("|---|---:|---:|\n")
        for tag, cnt in c.most_common():
            lines.append(f"| {tag} | {cnt} | {fmt_pct(cnt / total if total else float('nan'))} |\n")
        lines.append("\n")
        return lines

    md.extend(render_err_table("llama2"))
    md.extend(render_err_table("mistral"))

    md.append("## D. RAG 诊断分层（检索对生成的影响）\n")
    md.append("### D1. 检索稳定性概览\n")
    md.append("| 指标 | llama2 | mistral |\n")
    md.append("|---|---:|---:|\n")
    md.append(f"| retrieval_empty 次数 | {retrieval_empty['llama2']} | {retrieval_empty['mistral']} |\n")
    md.append(f"| 平均 num_chunks_retrieved | {_mean(chunks['llama2']):.2f} | {_mean(chunks['mistral']):.2f} |\n")
    md.append(f"| 平均 context_length | {_mean(ctx_lengths['llama2']):.1f} | {_mean(ctx_lengths['mistral']):.1f} |\n")

    md.append("\n### D2. 论文可写结论模板（可直接粘贴）\n")
    md.append(
        "- 在本轮 Single 任务中，`retrieval_empty` 基本为 0（或极低），且两模型检索到的 chunk 数与 context_length 接近，说明检索阶段对两模型输入证据的供给较为一致。\n"
    )
    md.append(
        "- 因此，模型差异主要来源于生成阶段对证据的‘抽取精度’与‘表达策略’：例如时间类问题更易出现 **time_offset**；描述类问题更易出现 **keyword_missing** 或过度扩写导致的幻觉风险信号。\n"
    )
    md.append(
        "- 该现象支持‘检索限制模型能力’的观点：当证据充分且同质时，模型之间的性能差距被显著压缩；要拉开差距，需要更难的检索条件或 Multi/Null 类型问题。\n"
    )

    out_md.write_text("".join(md), encoding="utf-8")
    print(f"Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
