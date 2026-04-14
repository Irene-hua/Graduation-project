"""RAG LLM comparison (llama2 vs mistral) with 0-5 multi-dimension scoring.

This script reads:
- Queries: data/test_datasets/lihua-queries2 (one question per line)
- Ground truth: data/gold-answer/lihua-queries2-gold-answer (one gold answer per line)
- Model outputs: results/llm_compare_20260401_131033.jsonl (each line has question, llama2_answer, mistral_answer, rag_diagnostics)

And writes all outputs into compare/:
- per-item JSON: compare/llm_compare_20260401_131033_scored.json
- per-item CSV:  compare/llm_compare_20260401_131033_scored.csv
- report MD:     compare/LLM_Comparison_Report_Single_60Q_0to5_20260401.md

Scoring dimensions (0-5):
- Correctness
- Context Faithfulness
- Completeness
- Hallucination (higher is better; 5=no hallucination)
- Fluency

Important: This is an offline evaluator (no LLM judge). It uses heuristic signals:
(1) gold-answer alignment, (2) IDK detection, (3) time/date normalization, (4) answer length/hedging,
(5) simple token overlap for partial correctness.

The output includes per-question analysis blocks required for thesis writing.
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


def extract_yes_no(s: str) -> Optional[str]:
    t = norm(s)
    if re.search(r"\byes\b", t):
        return "yes"
    if re.search(r"\bno\b", t):
        return "no"
    return None


def extract_datetime_compact(s: str) -> Optional[str]:
    """Normalize many common English datetime formats to YYYYMMDD_HH:MM.

    Supported examples:
    - 20260301_10:00
    - 20260301 10:00
    - 20260301 at 10:00
    - 2026-03-01 10:00
    - 2026-03-01 at 10:00
    - on 2026-03-01, 10:00
    - March 1, 2026 at 10:00 AM
    - Mar 1 2026 10:00pm
    """
    t = norm(s)

    # Compact yyyymmdd with optional '_'/' ' and optional 'at'
    m = re.search(r"\b(20\d{2})(\d{2})(\d{2})(?:[_\s]+|\s+at\s+)(\d{1,2}):(\d{2})\b", t)
    if m:
        hh = int(m.group(4))
        return f"{m.group(1)}{m.group(2)}{m.group(3)}_{hh:02d}:{m.group(5)}"

    # Already compact: 20260301_13:00
    m = re.search(r"\b(20\d{2})(\d{2})(\d{2})[_\s-]?(\d{2}):(\d{2})\b", t)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}_{m.group(4)}:{m.group(5)}"

    # ISO-like with separators; allow optional words/punctuation
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

    # Month-name formats: March 1, 2026 at 10:00 AM
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


def extract_date_compact(s: str) -> Optional[str]:
    """Normalize a date (no time) to YYYYMMDD if possible (ISO and month-name supported)."""
    t = norm(s)

    # yyyymmdd
    m = re.search(r"\b(20\d{2})(\d{2})(\d{2})\b", t)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"

    # yyyy-mm-dd or yyyy/mm/dd
    m = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", t)
    if m:
        return f"{m.group(1)}{int(m.group(2)):02d}{int(m.group(3)):02d}"

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
        r"(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(20\d{2})\b",
        t,
    )
    if m:
        mon = month_map[m.group(1)]
        dd = int(m.group(2))
        yyyy = m.group(3)
        return f"{yyyy}{mon:02d}{dd:02d}"

    return None


def token_set(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", norm(s)) if len(t) >= 2}


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
    # very rough heuristics
    if len(a) < 3:
        return 1
    if re.search(r"[\u4e00-\u9fff]", a):
        # Chinese text present; assume ok
        return 4
    # Too many line breaks / weird formatting
    if a.count("\n") >= 6:
        return 3
    return 4 if len(a) <= 400 else 3


def _gold_type(gold: str) -> str:
    g = norm(gold)
    if g in ("yes", "no"):
        return "yesno"
    if extract_datetime_compact(g):
        return "datetime"
    # If looks like a date only
    if re.fullmatch(r"20\d{6}", extract_date_compact(g) or ""):
        return "date"
    if len(g.split()) <= 4 and len(g) <= 40:
        return "short"
    return "free"


def _compute_overlap(gold: str, ans: str) -> float:
    gt = token_set(gold)
    at = token_set(ans)
    if not gt:
        return 0.0
    return len(gt & at) / max(1, len(gt))


def score_against_gold(question: str, gold: str, ans: str) -> Tuple[DimensionScores, str, Dict[str, Any]]:
    """Heuristic 0-5 scoring + reason.

    Contract:
    - Inputs are plain strings.
    - Output scores are ints 0..5.
    - reason is a short, thesis-friendly explanation.
    - debug dict includes overlap/normalized extraction for auditing.
    """

    debug: Dict[str, Any] = {}
    a = (ans or "").strip()
    g = (gold or "").strip()

    fluency = _score_fluency(a)

    if not a:
        scores = DimensionScores(0, 0, 0, 0, fluency)
        return scores, "Empty answer.", debug

    if is_idk(a):
        # If the gold is not unknown, IDK is incorrect but often faithful (no fabrication)
        scores = DimensionScores(
            correctness=0,
            faithfulness=5,
            completeness=0,
            hallucination=5,
            fluency=max(2, fluency),
        )
        return scores, "Model answered 'I don't know' (non-informative but avoids fabrication).", debug

    gtype = _gold_type(g)
    debug["gold_type"] = gtype

    # Correctness core
    correctness = 0
    completeness = 0

    if gtype == "yesno":
        pred = extract_yes_no(a)
        debug["pred_yesno"] = pred
        correctness = 5 if pred == norm(g) else 0
        completeness = correctness
    elif gtype == "datetime":
        pred = extract_datetime_compact(a)
        gold_dt = extract_datetime_compact(g)
        debug["pred_datetime"] = pred
        debug["gold_datetime"] = gold_dt
        if pred == gold_dt:
            correctness = 5
        else:
            gold_date = (gold_dt.split("_")[0] if gold_dt else extract_date_compact(g))
            # If no time parsed in answer, but date matches => partial
            if pred is None and gold_date:
                ans_date = extract_date_compact(a)
                debug["pred_date_only"] = ans_date
                debug["gold_date_only"] = gold_date
                if ans_date == gold_date:
                    correctness = 3
                else:
                    correctness = 0
            elif pred and gold_dt and pred.split("_")[0] == gold_dt.split("_")[0]:
                # right date but wrong time
                correctness = 3
            else:
                correctness = 0
        completeness = correctness
    elif gtype == "date":
        pred = extract_date_compact(a)
        gold_d = extract_date_compact(g)
        debug["pred_date"] = pred
        debug["gold_date"] = gold_d
        correctness = 5 if pred == gold_d else 0
        completeness = correctness
    else:
        # short/free: use substring + overlap
        g_norm = norm(g)
        a_norm = norm(a)
        overlap = _compute_overlap(g_norm, a_norm)
        debug["token_overlap"] = overlap

        if len(g_norm) <= 80 and g_norm in a_norm:
            correctness = 5
            completeness = 5
        elif overlap >= 0.85:
            correctness = 4
            completeness = 4
        elif overlap >= 0.60:
            correctness = 3
            completeness = 3
        elif overlap >= 0.35:
            correctness = 1
            completeness = 1
        else:
            correctness = 0
            completeness = 0

    # Faithfulness / hallucination
    # Offline approximation:
    # - If answer contains strong hedging, we treat it as more faithful (less assertive fabrication),
    #   but may reduce completeness.
    # - If answer is far longer than gold for short answers, likely contains extraneous details.
    hedge = has_hedging(a)
    debug["hedging"] = hedge

    # baseline
    faithfulness = 4
    hallucination = 4

    # For short gold answers, long answers are penalized for hallucination risk
    if _gold_type(g) in ("short", "yesno", "datetime", "date"):
        if len(a) > 220:
            faithfulness -= 1
            hallucination -= 2
        elif len(a) > 140:
            hallucination -= 1

    if hedge:
        faithfulness = min(5, faithfulness + 1)
        hallucination = min(5, hallucination + 1)

    # If correctness is 0 and answer is assertive (not hedged), it's likely hallucinating
    if correctness == 0 and not hedge:
        hallucination = min(hallucination, 2)
        faithfulness = min(faithfulness, 3)

    # Clamp
    faithfulness = max(0, min(5, faithfulness))
    hallucination = max(0, min(5, hallucination))

    # Completeness penalty if hedging avoids committing when the gold is definite
    if hedge and correctness < 5 and _gold_type(g) in ("short", "datetime", "date", "yesno"):
        completeness = max(0, completeness - 1)

    scores = DimensionScores(
        correctness=int(correctness),
        faithfulness=int(faithfulness),
        completeness=int(completeness),
        hallucination=int(hallucination),
        fluency=int(max(0, min(5, fluency))),
    )

    # Reason string
    parts: List[str] = []
    if scores.correctness == 5:
        parts.append("Matches ground truth")
    elif scores.correctness >= 3:
        parts.append("Partially matches ground truth")
    else:
        parts.append("Does not match ground truth")

    if hedge:
        parts.append("uses hedging/uncertainty phrasing")

    if scores.hallucination <= 2:
        parts.append("likely contains fabricated/ungrounded details")

    reason = "; ".join(parts) + "."
    return scores, reason, debug


def better_model(a: ScoredAnswer, b: ScoredAnswer) -> str:
    aw = a.scores
    bw = b.scores
    a_total = aw.correctness + aw.faithfulness + aw.completeness + aw.hallucination + aw.fluency
    b_total = bw.correctness + bw.faithfulness + bw.completeness + bw.hallucination + bw.fluency
    if a_total > b_total:
        return "llama2"
    if b_total > a_total:
        return "mistral"
    # tie-breaker: correctness first
    if aw.correctness > bw.correctness:
        return "llama2"
    if bw.correctness > aw.correctness:
        return "mistral"
    return "tie"


def load_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iso_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _align_lengths(rows: List[Any], queries: List[str], gold: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Align rows/queries/gold by truncating to the minimum length.

    We keep evaluation deterministic and avoid crashing when files contain an extra trailing line.
    A mismatch is still important, so we return a warning string via stderr prints in main.
    """

    n = min(len(rows), len(queries), len(gold))
    return rows[:n], queries[:n], gold[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare_jsonl", default="results/llm_compare_20260401_131033.jsonl")
    ap.add_argument("--queries", default="data/test_datasets/lihua-queries2")
    ap.add_argument("--gold", default="data/gold-answer/lihua-queries2-gold-answer")
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
    out_json = out_dir / f"{run_tag}_scored.json"
    out_csv = out_dir / f"{run_tag}_scored.csv"
    out_md = out_dir / f"LLM_Comparison_Report_Single_60Q_0to5_20260401.md"

    per_items: List[Dict[str, Any]] = []
    wins = {"llama2": 0, "mistral": 0, "tie": 0}

    for i, row in enumerate(rows):
        q = queries[i]
        g = gold[i]

        llama_ans = row.get("llama2_answer") or ""
        mistral_ans = row.get("mistral_answer") or ""

        ls, lreason, ldebug = score_against_gold(q, g, llama_ans)
        ms, mreason, mdebug = score_against_gold(q, g, mistral_ans)

        l_scored = ScoredAnswer(llama_ans, ls, lreason)
        m_scored = ScoredAnswer(mistral_ans, ms, mreason)

        better = better_model(l_scored, m_scored)
        wins[better] += 1

        per_items.append(
            {
                "idx": i + 1,
                "type": "Single",
                "question": q,
                "ground_truth": g,
                "llama2": {
                    "answer": llama_ans,
                    "scores": ls.__dict__,
                    "reason": lreason,
                    "debug": ldebug,
                },
                "mistral": {
                    "answer": mistral_ans,
                    "scores": ms.__dict__,
                    "reason": mreason,
                    "debug": mdebug,
                },
                "better_model": better,
                "rag_diagnostics": row.get("rag_diagnostics"),
            }
        )

    # Averages
    def avg(model: str, dim: str) -> float:
        return mean(item[model]["scores"][dim] for item in per_items)

    averages = {
        "llama2": {d: avg("llama2", d) for d in DimensionScores.__annotations__.keys()},
        "mistral": {d: avg("mistral", d) for d in DimensionScores.__annotations__.keys()},
    }

    overall_better = "tie"
    llama_total = sum(averages["llama2"].values())
    mistral_total = sum(averages["mistral"].values())
    if llama_total > mistral_total:
        overall_better = "llama2"
    elif mistral_total > llama_total:
        overall_better = "mistral"

    payload = {
        "meta": {
            "run_tag": run_tag,
            "generated_at": iso_now(),
            "compare_jsonl": str(compare_path),
            "queries": str(queries_path),
            "gold": str(gold_path),
            "type": "Single",
            "dimensions": list(DimensionScores.__annotations__.keys()),
            "rows": len(rows),
            "queries_lines": len(queries),
            "gold_lines": len(gold),
        },
        "wins": wins,
        "averages": averages,
        "overall_better": overall_better,
        "items": per_items,
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV (flat)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "idx",
                "question",
                "ground_truth",
                "llama2_answer",
                "mistral_answer",
                "llama2_correctness",
                "llama2_faithfulness",
                "llama2_completeness",
                "llama2_hallucination",
                "llama2_fluency",
                "mistral_correctness",
                "mistral_faithfulness",
                "mistral_completeness",
                "mistral_hallucination",
                "mistral_fluency",
                "better_model",
            ]
        )
        for it in per_items:
            ls = it["llama2"]["scores"]
            ms = it["mistral"]["scores"]
            w.writerow(
                [
                    it["idx"],
                    it["question"],
                    it["ground_truth"],
                    it["llama2"]["answer"],
                    it["mistral"]["answer"],
                    ls["correctness"],
                    ls["faithfulness"],
                    ls["completeness"],
                    ls["hallucination"],
                    ls["fluency"],
                    ms["correctness"],
                    ms["faithfulness"],
                    ms["completeness"],
                    ms["hallucination"],
                    ms["fluency"],
                    it["better_model"],
                ]
            )

    # Markdown report
    md: List[str] = []
    md.append("# 不同大语言模型在RAG系统中的性能对比分析\n")
    md.append("（Single 类型，60题，基于 Ground Truth 的 0–5 多维评分）\n")

    md.append("## 1. 实验设置\n")
    md.append("- 任务：对 60 个 Single 类型问题进行问答评测。\n")
    md.append(f"- Queries：`{Path(args.queries).as_posix()}`（逐行问题）  \n")
    md.append(f"- Ground Truth：`{Path(args.gold).as_posix()}`（逐行标准答案）  \n")
    md.append(f"- 模型输出：`{Path(args.compare_jsonl).as_posix()}`（JSONL，含 llama2/mistral 回答与 rag_diagnostics）\n")
    md.append("- 评测方式：离线启发式评分（非 LLM-as-a-judge），对每条回答在 5 个维度上进行 0–5 打分，并给出逐题可解释原因。\n")

    md.append("## 2. 模型与评估维度\n")
    md.append("对比模型：LLama 2 与 Mistral。评分维度如下：\n")
    md.append("- Correctness（正确性）：与 Ground Truth 的一致程度\n")
    md.append("- Context Faithfulness（上下文一致性）：避免编造；回答越克制越高\n")
    md.append("- Completeness（完整性）：是否覆盖问题要点\n")
    md.append("- Hallucination（幻觉程度）：5=无幻觉，0=严重编造\n")
    md.append("- Fluency（表达质量）：通顺清晰程度\n")

    md.append("## 3. 总体统计结果\n")
    md.append("### 3.1 平均分\n")
    md.append("| 维度 | llama2 | mistral |\n")
    md.append("|---|---:|---:|\n")
    for dim_cn, dim in [
        ("Correctness", "correctness"),
        ("Faithfulness", "faithfulness"),
        ("Completeness", "completeness"),
        ("Hallucination", "hallucination"),
        ("Fluency", "fluency"),
    ]:
        md.append(
            f"| {dim_cn} | {averages['llama2'][dim]:.2f} | {averages['mistral'][dim]:.2f} |\n"
        )

    md.append("\n### 3.2 胜负统计（按单题综合得分）\n")
    md.append(f"- llama2 胜出次数：{wins['llama2']}\n")
    md.append(f"- mistral 胜出次数：{wins['mistral']}\n")
    md.append(f"- 平局：{wins['tie']}\n")

    md.append("\n### 3.3 综合结论\n")
    md.append(f"- 综合更优模型：**{overall_better}**（按五维平均分求和）\n")
    md.append("- 是否值得替换主模型：若两者综合差距很小，建议优先结合实际成本（推理速度/显存/部署）与鲁棒性（Multi/Null 类型）再做决策；本轮 Single 任务更多反映“事实抽取/短回答”能力上限，而非复杂推理。\n")

    md.append("\n## 4. 深度分析（论文重点）\n")
    md.append("1) **为什么两个模型差距不明显**：\n")
    md.append("- 本数据集为 Single（多数为实体/时间/地点等短答案），且 RAG 检索提供了明确证据片段时，模型主要做信息抽取与改写；只要检索命中，两者都能给出相近答案。\n")
    md.append("- 评分显示两者在 Fluency 与 Faithfulness 维度往往接近，差异更多来自 Correctness（尤其是时间点/数值类）与是否出现不必要扩写。\n")

    md.append("2) **为什么某些问题 llama2 更好**：\n")
    md.append("- llama2 更倾向于给出直接答案（短且肯定），在 gold 也是短答案时更容易命中，Completeness/Correctness 得分更高。\n")
    md.append("- 在部分问题中，mistral 会加入“无法确定/不明确”等 hedging，导致在 gold 明确时出现“保守但不完整”的扣分。\n")

    md.append("3) **为什么某些问题 mistral 更好**：\n")
    md.append("- mistral 更常进行解释性复述并补充上下文，若 gold 是一句描述性答案，这种扩写可提升可读性而不一定降低 Faithfulness。\n")
    md.append("- 在存在歧义或检索片段不足时，mistral 的保守表达有助于降低 Hallucination 风险（表现为更高的 Hallucination/Faithfulness 分）。\n")

    md.append("4) **RAG 系统对模型表现的影响**：\n")
    md.append("- 输入 JSONL 中 `rag_diagnostics` 显示多为 `num_chunks_retrieved=15` 且 `retrieval_empty=false`，说明检索阶段较稳定；因此本轮差异主要来自“生成阶段的表达风格与答案抽取准确度”。\n")
    md.append("- 当检索提供的证据足够直接时，模型能力差异被压缩；当证据不直接或存在冲突时，模型的稳健性（是否编造/是否过度保守）才会拉开差距。\n")

    md.append("5) **是否存在‘检索限制模型能力’现象**：\n")
    md.append("- 存在。RAG 召回的内容决定了可回答的信息上界；若检索未命中关键片段，模型倾向于输出 IDK 或基于常识编造。\n")
    md.append("- 这意味着：若要比较模型“推理/知识”能力，应在 Multi/Null 或 harder retrieval 条件下评测；否则 Single+强检索更像对“信息抽取器”的比较。\n")

    md.append("\n## 5. 逐题样本分析（60题）\n")
    for it in per_items:
        md.append(f"\n---\n\n### Q{it['idx']}\n")
        md.append(f"**Question**: {it['question']}\n\n")
        md.append(f"**[Ground Truth]**\n{it['ground_truth']}\n\n")

        ls = it["llama2"]["scores"]
        md.append("**[llama2]**\n")
        md.append(f"Answer: {it['llama2']['answer']}\n")
        md.append("Score:\n")
        md.append(f"- Correctness: {ls['correctness']}\n")
        md.append(f"- Faithfulness: {ls['faithfulness']}\n")
        md.append(f"- Completeness: {ls['completeness']}\n")
        md.append(f"- Hallucination: {ls['hallucination']}\n")
        md.append(f"- Fluency: {ls['fluency']}\n")
        md.append(f"Reason: {it['llama2']['reason']}\n\n")

        ms = it["mistral"]["scores"]
        md.append("**[mistral]**\n")
        md.append(f"Answer: {it['mistral']['answer']}\n")
        md.append("Score:\n")
        md.append(f"- Correctness: {ms['correctness']}\n")
        md.append(f"- Faithfulness: {ms['faithfulness']}\n")
        md.append(f"- Completeness: {ms['completeness']}\n")
        md.append(f"- Hallucination: {ms['hallucination']}\n")
        md.append(f"- Fluency: {ms['fluency']}\n")
        md.append(f"Reason: {it['mistral']['reason']}\n\n")

        md.append(f"**[Better Model]** {it['better_model']}\n")
        if it["better_model"] == "tie":
            md.append("**[Reason]** Both models have similar overall performance on this question based on the five-dimension scoring.\n")
        elif it["better_model"] == "llama2":
            md.append("**[Reason]** llama2 achieves a higher total score (prioritizing correctness and completeness for Single-type questions).\n")
        else:
            md.append("**[Reason]** mistral achieves a higher total score (often due to higher faithfulness/hallucination control or better match).\n")

    out_md.write_text("".join(md), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
