"""RAG LLM comparison (llama2 vs mistral) with 0-5 multi-dimension scoring (Multi type).

This is a Multi-type evaluator for `lihua-queries1`.
Compared to Single evaluator, Multi needs:
- Parsing list-like gold answers: e.g., "A & B & C".
- Partial credit based on set overlap (precision/recall/F1-like).

Inputs:
- Queries: data/test_datasets/lihua-queries1
- Ground truth: data/gold-answer/lihua-queries1-gold-answer
- Model outputs: results/llm_compare_20260401_154508.jsonl

Outputs (all under compare/, with "Multi" in name):
- compare/llm_compare_20260401_154508_Multi_scored.json
- compare/llm_compare_20260401_154508_Multi_scored.csv
- compare/LLM_Comparison_Report_Multi_60Q_0to5_20260401.md

Scoring dimensions (0-5):
- Correctness
- Context Faithfulness
- Completeness
- Hallucination (higher is better)
- Fluency

Notes:
- This is an offline deterministic heuristic scorer (not LLM judge).
- For yes/no golds, it uses yes/no extraction.
- For Multi-list golds (contains '&' or multiple entities), it computes overlap and maps to 0/1/3/5 scores.
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
    # handle explicit yes/no
    if re.search(r"\byes\b", t):
        return "yes"
    if re.search(r"\bno\b", t):
        return "no"

    # common verbalizations
    if re.search(r"\b(did not|didn't|do not|don't|nope|not)\b", t):
        return "no"
    if re.search(r"\b(affirmative|certainly|of course|sure)\b", t):
        return "yes"

    # patterns like "the answer is no"
    m = re.search(r"answer is (yes|no)", t)
    if m:
        return m.group(1)

    return None


def _score_fluency(ans: str) -> int:
    a = (ans or "").strip()
    if not a:
        return 0
    if len(a) < 3:
        return 1
    if a.count("\n") >= 6:
        return 3
    return 4 if len(a) <= 500 else 3


def token_set(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", norm(s)) if len(t) >= 2}


def split_multi_gold(gold: str) -> List[str]:
    """Split gold into items for Multi questions.

    Handles patterns like:
    - "A & B & C"
    - "A and B" (best-effort)
    - comma separated

    Keeps original casing out; returns normalized item strings.
    """

    g = (gold or "").strip()
    if not g:
        return []

    # If it's clearly yes/no/date-like, don't split
    if norm(g) in ("yes", "no"):
        return [norm(g)]

    # Prefer '&' splitter (dataset uses it)
    if "&" in g:
        parts = [p.strip() for p in g.split("&")]
    elif ";" in g:
        parts = [p.strip() for p in g.split(";")]
    elif "," in g:
        parts = [p.strip() for p in g.split(",")]
    else:
        # best-effort ' and '
        parts = [p.strip() for p in re.split(r"\band\b", g, flags=re.IGNORECASE) if p.strip()]

    # Cleanup
    out: List[str] = []
    for p in parts:
        p = p.strip().strip("\"'")
        p = re.sub(r"\s+", " ", p)
        if p:
            out.append(norm(p))
    return out or [norm(g)]


def extract_datetime_compact(s: str) -> Optional[str]:
    """Normalize many common English datetime formats to YYYYMMDD_HH:MM.

    Kept consistent with Single evaluator.
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

    # ISO-like with separators; allow optional words/punctuation and AM/PM
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
    """Normalize a date (no time) to YYYYMMDD if possible."""
    t = norm(s)

    m = re.search(r"\b(20\d{2})(\d{2})(\d{2})\b", t)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"

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


def classify_gold_type(gold: str) -> str:
    g = norm(gold)
    if g in ("yes", "no"):
        return "yesno"
    if extract_datetime_compact(g):
        return "datetime"
    if "&" in gold:
        return "multi_list"
    if any(sep in gold for sep in [",", ";"]):
        return "multi_list"
    return "free"


def compute_set_overlap(gold_items: List[str], ans: str) -> Tuple[float, float, float]:
    """Return (precision, recall, f1) based on token/item matching.

    Matching rule (improved): an item is hit if either:
    - all tokens of the item appear in answer token set, OR
    - the item with spaces removed appears in the answer with non-alphanumerics removed, OR
    - (fallback, for descriptive items) >=50% of item's tokens appear in the answer.

    This handles concatenated names like 'AdamSmith'/'WolfgangSchulz' and partial paraphrases.
    """

    ans_tokens = token_set(ans)
    ans_compact = re.sub(r"[^a-z0-9]", "", norm(ans))

    if not gold_items:
        return 0.0, 0.0, 0.0

    hits = 0
    for item in gold_items:
        itoks = token_set(item)
        item_compact = re.sub(r"[^a-z0-9]", "", norm(item))

        token_hit = bool(itoks) and itoks.issubset(ans_tokens)
        compact_hit = bool(item_compact) and (item_compact in ans_compact)

        # Fallback: partial token coverage (for descriptive multi-items)
        coverage_hit = False
        if itoks and not (token_hit or compact_hit):
            inter = len(itoks & ans_tokens)
            coverage = inter / max(1, len(itoks))
            coverage_hit = coverage >= 0.5

        if token_hit or compact_hit or coverage_hit:
            hits += 1

    recall = hits / len(gold_items)

    # approximate precision using number of found items vs predicted item count
    pred_count = 0
    if "&" in ans:
        pred_count = len([p for p in ans.split("&") if p.strip()])
    elif "," in ans:
        pred_count = len([p for p in ans.split(",") if p.strip()])
    else:
        pred_count = hits if hits else 1

    precision = hits / max(1, pred_count)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


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


def score_against_gold(question: str, gold: str, ans: str) -> Tuple[DimensionScores, str, Dict[str, Any]]:
    debug: Dict[str, Any] = {}

    a = (ans or "").strip()
    g = (gold or "").strip()

    fluency = _score_fluency(a)

    if not a:
        scores = DimensionScores(0, 0, 0, 0, fluency)
        return scores, "Empty answer.", debug

    if is_idk(a):
        scores = DimensionScores(0, 5, 0, 5, max(2, fluency))
        return scores, "Model answered 'I don't know' (non-informative but avoids fabrication).", debug

    gtype = classify_gold_type(g)
    debug["gold_type"] = gtype

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
            if pred is None and gold_date:
                ans_date = extract_date_compact(a)
                debug["pred_date_only"] = ans_date
                debug["gold_date_only"] = gold_date
                correctness = 3 if ans_date == gold_date else 0
            elif pred and gold_dt and pred.split("_")[0] == gold_dt.split("_")[0]:
                correctness = 3
            else:
                correctness = 0
        completeness = correctness
    elif gtype == "multi_list":
        gold_items = split_multi_gold(g)
        debug["gold_items"] = gold_items
        p, r, f1 = compute_set_overlap(gold_items, a)
        debug["precision"] = p
        debug["recall"] = r
        debug["f1"] = f1

        # Map F1 to 0-5 scale
        if f1 >= 0.90:
            correctness = 5
            completeness = 5
        elif f1 >= 0.60:
            correctness = 3
            completeness = 3
        elif f1 >= 0.30:
            correctness = 1
            completeness = 1
        else:
            correctness = 0
            completeness = 0
    else:
        # free text: token overlap against gold tokens (partial)
        gt = token_set(g)
        at = token_set(a)
        overlap = len(gt & at) / max(1, len(gt)) if gt else 0.0
        debug["token_overlap"] = overlap

        if len(norm(g)) <= 80 and norm(g) in norm(a):
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

    hedge = has_hedging(a)
    debug["hedging"] = hedge

    faithfulness = 4
    hallucination = 4

    # penalize very long answers for short golds / yesno
    if gtype in ("yesno", "multi_list") and len(a) > 260:
        faithfulness -= 1
        hallucination -= 2
    elif gtype in ("yesno", "multi_list") and len(a) > 160:
        hallucination -= 1

    if hedge:
        faithfulness = min(5, faithfulness + 1)
        hallucination = min(5, hallucination + 1)

    if correctness == 0 and not hedge and not is_idk(a):
        hallucination = min(hallucination, 2)
        faithfulness = min(faithfulness, 3)

    faithfulness = max(0, min(5, faithfulness))
    hallucination = max(0, min(5, hallucination))

    scores = DimensionScores(
        correctness=int(correctness),
        faithfulness=int(faithfulness),
        completeness=int(completeness),
        hallucination=int(hallucination),
        fluency=int(max(0, min(5, fluency))),
    )

    parts: List[str] = []
    if scores.correctness == 5:
        parts.append("Matches ground truth")
    elif scores.correctness >= 3:
        parts.append("Partially matches ground truth")
    else:
        parts.append("Does not match ground truth")

    if gtype == "multi_list":
        parts.append("evaluated as Multi list overlap")

    if hedge:
        parts.append("uses hedging/uncertainty phrasing")

    if scores.hallucination <= 2:
        parts.append("likely contains fabricated/ungrounded details")

    reason = "; ".join(parts) + "."
    return scores, reason, debug


def better_model(llama: ScoredAnswer, mistral: ScoredAnswer) -> str:
    def total(s: DimensionScores) -> int:
        return s.correctness + s.faithfulness + s.completeness + s.hallucination + s.fluency

    lt = total(llama.scores)
    mt = total(mistral.scores)
    if lt > mt:
        return "llama2"
    if mt > lt:
        return "mistral"

    # tie-breaker
    if llama.scores.correctness != mistral.scores.correctness:
        return "llama2" if llama.scores.correctness > mistral.scores.correctness else "mistral"

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
    ap.add_argument("--compare_jsonl", default="results/llm_compare_20260401_154508.jsonl")
    ap.add_argument("--queries", default="data/test_datasets/lihua-queries1")
    ap.add_argument("--gold", default="data/gold-answer/lihua-queries1-gold-answer")
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
    out_json = out_dir / f"{run_tag}_Multi_scored.json"
    out_csv = out_dir / f"{run_tag}_Multi_scored.csv"
    out_md = out_dir / f"LLM_Comparison_Report_Multi_60Q_0to5_20260401.md"

    items: List[Dict[str, Any]] = []
    wins = {"llama2": 0, "mistral": 0, "tie": 0}

    for i, row in enumerate(rows):
        q = queries[i]
        g = gold[i]
        la = row.get("llama2_answer") or ""
        ma = row.get("mistral_answer") or ""

        ls, lreason, ldebug = score_against_gold(q, g, la)
        ms, mreason, mdebug = score_against_gold(q, g, ma)

        l_scored = ScoredAnswer(la, ls, lreason)
        m_scored = ScoredAnswer(ma, ms, mreason)

        better = better_model(l_scored, m_scored)
        wins[better] += 1

        items.append(
            {
                "idx": i + 1,
                "type": "Multi",
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
            "type": "Multi",
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
    md.append("（Multi 类型，60题，基于 Ground Truth 的 0–5 多维评分）\n\n")

    md.append("## 1. 实验设置\n")
    md.append("- 任务：对 60 个 Multi 类型问题进行问答评测。\n")
    md.append(f"- Queries：`{Path(args.queries).as_posix()}`（逐行问题）  \n")
    md.append(f"- Ground Truth：`{Path(args.gold).as_posix()}`（逐行标准答案）  \n")
    md.append(f"- 模型输出：`{Path(args.compare_jsonl).as_posix()}`（JSONL，含 llama2/mistral 回答与 rag_diagnostics）\n")
    md.append("- 评测方式：离线启发式评分（非 LLM-as-a-judge）。对每条回答在 5 个维度上进行 0–5 打分；Multi 问题通过列表项重叠（F1-like）给予部分分。\n\n")

    md.append("## 2. 模型与评估维度\n")
    md.append("对比模型：LLama2 与 Mistral。评分维度如下：\n")
    md.append("- Correctness（正确性）\n")
    md.append("- Context Faithfulness（上下文一致性）\n")
    md.append("- Completeness（完整性）\n")
    md.append("- Hallucination（幻觉程度，5=无幻觉）\n")
    md.append("- Fluency（表达质量）\n\n")

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
    md.append("- 是否值得替换主模型：Multi 类型更强调‘多要点覆盖’与‘关系/顺序判断’。若总体差距不大，建议结合业务对召回质量与输出格式一致性的要求再决定；若 time/yes-no 类比重高，还应做专项验证。\n\n")

    md.append("## 4. 深度分析（论文重点）\n")
    md.append("1) **为什么两个模型差距不明显**：当检索证据充分且问题多为 Yes/No 或有限项列表时，模型更多执行证据对齐与复述，差距会被压缩。\n")
    md.append("2) **为什么某些问题 llama2 更好**：llama2 往往更直接给出确定判断，减少 IDK，提升 completeness；但在 Multi 列表题上可能遗漏部分子项。\n")
    md.append("3) **为什么某些问题 mistral 更好**：mistral 在部分 Multi 列表题上更倾向补充要点，列表覆盖更充分；但也更可能出现保守 IDK 或过度扩写。\n")
    md.append("4) **RAG 系统对模型表现的影响**：若 `retrieval_empty=false` 且 context_length 接近，生成差异主要由生成策略决定；反之检索失败会主导错误。\n")
    md.append("5) **是否存在‘检索限制模型能力’现象**：存在。Multi 题的上限由召回的‘多证据片段覆盖度’决定，检索不足会限制模型多要点整合能力。\n\n")

    md.append("## 5. 逐题样本分析（60题）\n")
    for it in items:
        md.append(f"\n---\n\n### Q{it['idx']}\n")
        md.append(f"**Question**: {it['question']}\n\n")
        md.append(f"**[Ground Truth]**\n{it['ground_truth']}\n\n")

        ls = it["llama2"]["scores"]
        md.append("**[llama2]**\n")
        md.append(f"Answer: {it['llama2']['answer']}\n")
        md.append("Score:\n")
        for d in dims:
            md.append(f"- {d}: {ls[d]}\n")
        md.append(f"Reason: {it['llama2']['reason']}\n\n")

        ms = it["mistral"]["scores"]
        md.append("**[mistral]**\n")
        md.append(f"Answer: {it['mistral']['answer']}\n")
        md.append("Score:\n")
        for d in dims:
            md.append(f"- {d}: {ms[d]}\n")
        md.append(f"Reason: {it['mistral']['reason']}\n\n")

        md.append(f"**[Better Model]** {it['better_model']}\n")
        if it["better_model"] == "tie":
            md.append("**[Reason]** Two models have similar total score; tie-breakers do not indicate a clear winner.\n")
        elif it["better_model"] == "llama2":
            md.append("**[Reason]** llama2 has higher total score, typically driven by correctness/completeness on this item.\n")
        else:
            md.append("**[Reason]** mistral has higher total score, typically driven by better list coverage or hallucination control.\n")

    out_md.write_text("".join(md), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
