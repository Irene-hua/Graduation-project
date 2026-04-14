"""RAG LLM comparison (llama2 vs mistral) with 0-5 multi-dimension scoring (Null type).

Null-type dataset contract (lihua-queries3):
- Each question is intended to be unanswerable from retrieved context.
- Ground truth file uses the phrase "Insufficient information" (one line per question).

Therefore the *primary* skill is abstention: the best answer is a faithful refusal (IDK) that
explicitly indicates missing context, without inventing details.

Reads:
- Queries: data/test_datasets/lihua-queries3
- Ground truth: data/gold-answer/lihua-queries3-gold-answer
- Model outputs: results/llm_compare_20260401_184038.jsonl

Writes (all under compare/, with "Null" in name):
- compare/llm_compare_20260401_184038_Null_scored.json
- compare/llm_compare_20260401_184038_Null_scored.csv
- compare/LLM_Comparison_Report_Null_60Q_0to5_20260401.md

Scoring dimensions (0-5):
- correctness: whether the model appropriately abstains (5) vs hallucinates (0)
- faithfulness: whether it avoids making claims not supported by context (Null => high for IDK)
- completeness: whether it clearly signals unanswerable / missing info (5 for good refusal)
- hallucination: 5=no fabricated details; 0=severe fabrication (numbers/dates/entities asserted)
- fluency: readability

This evaluator is deterministic (no LLM judge).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


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


def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def is_idk(ans: str) -> bool:
    a = norm(ans)
    return any(p in a for p in IDK_PATTERNS)


def has_hedging(ans: str) -> bool:
    a = norm(ans)
    return any(p in a for p in HEDGE_PATTERNS)


def token_set(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", norm(s)) if len(t) >= 2}


def extract_datetime_compact(s: str) -> Optional[str]:
    """Aggressive datetime normalization to YYYYMMDD_HH:MM (aligned with Single/Multi)."""
    t = norm(s)

    m = re.search(r"\b(20\d{2})(\d{2})(\d{2})(?:[_\s]+|\s+at\s+)(\d{1,2}):(\d{2})\b", t)
    if m:
        hh = int(m.group(4))
        return f"{m.group(1)}{m.group(2)}{m.group(3)}_{hh:02d}:{m.group(5)}"

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


def has_specifics(ans: str) -> bool:
    """Detect potentially hallucinated specifics: timestamps, dates, money, percentages, IDs.

    This is a conservative signal: if present in a Null question, it's risky.
    """
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
    if re.search(r"\b(id|invoice|order)\s*#?\s*\d+\b", t):
        return True
    return False


@dataclass
class DimensionScores:
    correctness: int
    faithfulness: int
    completeness: int
    hallucination: int
    fluency: int


@dataclass
class ScoredAnswer:
    answer: str
    scores: DimensionScores
    reason: str


def _score_fluency(ans: str) -> int:
    a = (ans or "").strip()
    if not a:
        return 0
    if len(a) < 3:
        return 1
    if a.count("\n") >= 6:
        return 3
    return 4 if len(a) <= 450 else 3


def score_null_against_gold(question: str, gold: str, ans: str) -> Tuple[DimensionScores, str, Dict[str, Any]]:
    """Null scoring contract:

    - Gold is expected to be an abstention marker (e.g., 'Insufficient information').
    - A good answer should explicitly refuse and indicate missing info.
    - Any asserted specifics are treated as hallucination risk.

    Returns: (scores, reason, debug)
    """

    debug: Dict[str, Any] = {}
    a = (ans or "").strip()
    g = (gold or "").strip()

    fluency = _score_fluency(a)
    if not a:
        scores = DimensionScores(0, 0, 0, 0, fluency)
        return scores, "Empty answer.", debug

    an = norm(a)

    # Accept both explicit IDK and explicit 'missing info' statements as abstention.
    idk = is_idk(a)
    missing_info_signal = any(
        p in an
        for p in (
            "context does not provide",
            "does not provide information",
            "doesn't provide information",
            "does not provide any information",
            "doesn't provide any information",
            "no information available",
            "it is not possible to determine",
            "not possible to determine",
            "cannot answer",
            "can't answer",
            "i cannot answer",
            "i can't answer",
            "not provided",
            "not specified",
            "not mentioned",
            "not stated",
            "no information",
            "insufficient information",
            "cannot determine",
            "can't determine",
            "unable to determine",
        )
    )
    abstained = idk or missing_info_signal

    hedge = has_hedging(a)
    specifics = has_specifics(a)
    debug.update(
        {
            "gold": g,
            "idk": idk,
            "missing_info_signal": missing_info_signal,
            "abstained": abstained,
            "hedging": hedge,
            "has_specifics": specifics,
        }
    )

    # Base faithfulness/hallucination (Null favors abstention)
    faithfulness = 3
    hallucination = 3

    if abstained:
        faithfulness = 5
        hallucination = 5
    else:
        # Not refusing in Null is risky
        faithfulness = 2
        hallucination = 2

    if specifics and not abstained:
        hallucination = 0
        faithfulness = min(faithfulness, 1)

    if hedge and abstained:
        faithfulness = 5
        hallucination = 5

    # Correctness/completeness focus on abstention
    if abstained:
        correctness = 5
        completeness = 5
    else:
        correctness = 0
        completeness = 0

    # Clamp
    scores = DimensionScores(
        correctness=int(max(0, min(5, correctness))),
        faithfulness=int(max(0, min(5, faithfulness))),
        completeness=int(max(0, min(5, completeness))),
        hallucination=int(max(0, min(5, hallucination))),
        fluency=int(max(0, min(5, fluency))),
    )

    if scores.correctness == 5:
        if idk:
            reason = "Correct abstention for Null question (explicit IDK / insufficient information)."
        else:
            reason = "Correct abstention for Null question (explicit missing-information statement)."
    else:
        reason = "Did not abstain for Null question; likely hallucinated or over-assertive."

    if scores.hallucination <= 2:
        reason += " Hallucination risk detected."

    return scores, reason, debug


def better_model(llama: ScoredAnswer, mistral: ScoredAnswer) -> str:
    def total(s: DimensionScores) -> int:
        return s.correctness + s.faithfulness + s.completeness + s.hallucination + s.fluency

    # Null-specific: if both answers are perfect abstentions, treat as tie (ignore small fluency differences).
    def is_perfect_abstention(s: DimensionScores) -> bool:
        return (
            s.correctness == 5
            and s.faithfulness == 5
            and s.completeness == 5
            and s.hallucination == 5
        )

    if is_perfect_abstention(llama.scores) and is_perfect_abstention(mistral.scores):
        return "tie"

    # Null-specific: if both models have the same core outcome (both right or both wrong),
    # don't let small fluency differences decide the winner.
    core_dims = ("correctness", "faithfulness", "completeness", "hallucination")
    if all(getattr(llama.scores, d) == getattr(mistral.scores, d) for d in core_dims):
        return "tie"

    lt = total(llama.scores)
    mt = total(mistral.scores)
    if lt > mt:
        return "llama2"
    if mt > lt:
        return "mistral"

    # tie-breaker: abstention correctness first, then hallucination
    if llama.scores.correctness != mistral.scores.correctness:
        return "llama2" if llama.scores.correctness > mistral.scores.correctness else "mistral"
    if llama.scores.hallucination != mistral.scores.hallucination:
        return "llama2" if llama.scores.hallucination > mistral.scores.hallucination else "mistral"

    return "tie"


def load_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iso_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _align_lengths(rows: List[Any], queries: List[str], gold: List[str]) -> Tuple[List[Any], List[str], List[str]]:
    n = min(len(rows), len(queries), len(gold))
    return rows[:n], queries[:n], gold[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare_jsonl", default="results/llm_compare_20260401_184038.jsonl")
    ap.add_argument("--queries", default="data/test_datasets/lihua-queries3")
    ap.add_argument("--gold", default="data/gold-answer/lihua-queries3-gold-answer")
    ap.add_argument("--out_dir", default="compare")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    compare_path = (root / args.compare_jsonl).resolve()
    queries_path = (root / args.queries).resolve()
    gold_path = (root / args.gold).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(compare_path)
    queries = load_lines(queries_path)
    gold = load_lines(gold_path)

    if not (len(rows) == len(queries) == len(gold)):
        print(
            f"[warn] Length mismatch detected; will truncate to min length. "
            f"rows={len(rows)} queries={len(queries)} gold={len(gold)}"
        )
        rows, queries, gold = _align_lengths(rows, queries, gold)

    run_tag = compare_path.stem
    out_json = out_dir / f"{run_tag}_Null_scored.json"
    out_csv = out_dir / f"{run_tag}_Null_scored.csv"
    out_md = out_dir / f"LLM_Comparison_Report_Null_60Q_0to5_20260401.md"

    items: List[Dict[str, Any]] = []
    wins = {"llama2": 0, "mistral": 0, "tie": 0}

    for i, row in enumerate(rows):
        q = queries[i]
        g = gold[i]
        la = row.get("llama2_answer") or ""
        ma = row.get("mistral_answer") or ""

        ls, lreason, ldebug = score_null_against_gold(q, g, la)
        ms, mreason, mdebug = score_null_against_gold(q, g, ma)

        l_scored = ScoredAnswer(la, ls, lreason)
        m_scored = ScoredAnswer(ma, ms, mreason)

        better = better_model(l_scored, m_scored)
        wins[better] += 1

        items.append(
            {
                "idx": i + 1,
                "type": "Null",
                "question": q,
                "ground_truth": g,
                "llama2": {"answer": la, "scores": ls.__dict__, "reason": lreason, "debug": ldebug},
                "mistral": {"answer": ma, "scores": ms.__dict__, "reason": mreason, "debug": mdebug},
                "better_model": better,
                "rag_diagnostics": row.get("rag_diagnostics"),
            }
        )

    dims = list(DimensionScores.__annotations__.keys())

    def avg(model: str, dim: str) -> float:
        return mean(it[model]["scores"][dim] for it in items)

    averages = {
        "llama2": {d: avg("llama2", d) for d in dims},
        "mistral": {d: avg("mistral", d) for d in dims},
    }

    overall_better = "tie"
    if sum(averages["llama2"].values()) > sum(averages["mistral"].values()):
        overall_better = "llama2"
    elif sum(averages["mistral"].values()) > sum(averages["llama2"].values()):
        overall_better = "mistral"

    payload = {
        "meta": {
            "run_tag": run_tag,
            "generated_at": iso_now(),
            "compare_jsonl": str(compare_path),
            "queries": str(queries_path),
            "gold": str(gold_path),
            "type": "Null",
            "dimensions": dims,
            "rows": len(rows),
            "queries_lines": len(queries),
            "gold_lines": len(gold),
        },
        "wins": wins,
        "averages": averages,
        "overall_better": overall_better,
        "items": items,
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "idx",
                "question",
                "ground_truth",
                "llama2_answer",
                "mistral_answer",
                *(f"llama2_{d}" for d in dims),
                *(f"mistral_{d}" for d in dims),
                "better_model",
            ]
        )
        for it in items:
            ls = it["llama2"]["scores"]
            ms = it["mistral"]["scores"]
            w.writerow(
                [
                    it["idx"],
                    it["question"],
                    it["ground_truth"],
                    it["llama2"]["answer"],
                    it["mistral"]["answer"],
                    *(ls[d] for d in dims),
                    *(ms[d] for d in dims),
                    it["better_model"],
                ]
            )

    # Markdown report
    md: List[str] = []
    md.append("# 不同大语言模型在RAG系统中的性能对比分析\n")
    md.append("（Null 类型，60题，基于 Ground Truth 的 0–5 多维评分）\n\n")

    md.append("## 1. 实验设置\n")
    md.append("- 任务：对 60 个 Null 类型问题进行问答评测（问题刻意设计为无法从检索上下文中得到唯一答案）。\n")
    md.append(f"- Queries：`{Path(args.queries).as_posix()}`（逐行问题）  \n")
    md.append(f"- Ground Truth：`{Path(args.gold).as_posix()}`（逐行标准答案，均为 Insufficient information）  \n")
    md.append(f"- 模型输出：`{Path(args.compare_jsonl).as_posix()}`（JSONL，含 llama2/mistral 回答与 rag_diagnostics）\n")
    md.append("- 评测方式：离线启发式评分（非 LLM-as-a-judge）。Null 任务以‘正确拒答/避免幻觉’为主要目标。\n\n")

    md.append("## 2. 模型与评估维度\n")
    md.append("对比模型：LLama2 与 Mistral。评分维度如下：\n")
    md.append("- Correctness（正确性）：是否正确拒答/表明信息不足（Null 的正确答案）\n")
    md.append("- Context Faithfulness（上下文一致性）：是否避免编造\n")
    md.append("- Completeness（完整性）：拒���是否清晰、是否说明缺失信息\n")
    md.append("- Hallucination（幻觉程度）：5=无幻觉，0=严重编造（如编造日期/金额/具体事实）\n")
    md.append("- Fluency（表达质量）：表达通顺清晰\n\n")

    md.append("## 3. 总体统计结果\n")
    md.append("### 3.1 平均分\n")
    md.append("| 维度 | llama2 | mistral |\n")
    md.append("|---|---:|---:|\n")
    for dim in dims:
        md.append(f"| {dim} | {averages['llama2'][dim]:.2f} | {averages['mistral'][dim]:.2f} |\n")

    md.append("\n### 3.2 胜负统计（按单题综合得分）\n")
    md.append(f"- llama2 胜出次数：{wins['llama2']}\n")
    md.append(f"- mistral 胜出次数：{wins['mistral']}\n")
    md.append(f"- 平局：{wins['tie']}\n\n")

    md.append("### 3.3 综合结论\n")
    md.append(f"- 综合更优模型：**{overall_better}**（按五维平均分求和）\n")
    md.append(
        "- 是否值得替换主模型：Null 任务衡量的是‘安全拒答与幻觉控制’能力。若某模型在 Hallucination 与 Faithfulness 显著更高，则更适合作为 RAG 的默认回答器（尤其在检索空/证据不足时）。\n\n"
    )

    md.append("## 4. 深度分析（论文重点）\n")
    md.append("1) **为什么两个模型差距不明显**：当两者都能识别‘信息不足’并选择拒答时，得分会集中在高分区间；差异主要来自少数题中是否出现具体编造。\n")
    md.append("2) **为什么某些问题 llama2 更好**：llama2 若更频繁地给出明确拒答（IDK）且不扩写，将在 Correctness/Hallucination 维度占优。\n")
    md.append("3) **为什么某些问题 mistral 更好**：mistral 若更倾向解释‘缺少哪些信息’，在 Completeness/Faithfulness 上更稳定；但若扩写带入具体细节，会被幻觉惩罚。\n")
    md.append("4) **RAG 系统对模型表现的影响**：Null 数据用于模拟检索证据不足的情形，即使检索返回了片段，片段也无法支持问题的精确回答。此时模型是否选择拒答比语言能力更关键。\n")
    md.append("5) **是否存在‘检索限制模型能力’现象**：存在。RAG 召回决定可回答信息上界；在 Null 条件下，上界为‘不可回答’，模型应避免用常识补全。\n\n")

    md.append("## 5. 逐题样本分析（60题）\n")
    for it in items:
        md.append(f"\n---\n\n### Q{it['idx']}\n")
        md.append(f"**Question**: {it['question']}\n\n")
        md.append(f"**[Ground Truth]**\n{it['ground_truth']}\n\n")

        for model in ("llama2", "mistral"):
            md.append(f"**[{model}]**\n")
            md.append(f"Answer: {it[model]['answer']}\n")
            md.append("Score:\n")
            for d in dims:
                md.append(f"- {d}: {it[model]['scores'][d]}\n")
            md.append(f"Reason: {it[model]['reason']}\n\n")

        md.append(f"**[Better Model]** {it['better_model']}\n")
        if it["better_model"] == "tie":
            md.append("**[Reason]** Both models show similar abstention/hallucination behavior on this item.\n")
        elif it["better_model"] == "llama2":
            md.append("**[Reason]** llama2 shows better abstention correctness and/or lower hallucination risk.\n")
        else:
            md.append("**[Reason]** mistral shows better abstention correctness and/or lower hallucination risk.\n")

    out_md.write_text("".join(md), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
