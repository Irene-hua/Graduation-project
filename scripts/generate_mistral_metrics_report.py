"""Generate mistral-only metrics report from scored CSVs.

This script is designed for Windows PowerShell users: you run it as a normal
Python module, not by pasting Python code into the PowerShell prompt.

It extracts the mistral model rows from three scored CSVs (Single/Multi/Null),
counts:
- correct
- wrong
- unanswered (abstention / "I don't know" / not found)

and computes Precision/Recall/F1 under a simple, explicit definition.

Usage (PowerShell):
  python -m scripts.generate_mistral_metrics_report

Optional:
  python -m scripts.generate_mistral_metrics_report --threshold 5

Outputs:
  compare/mistral_metrics_report_YYYYMMDD.md
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TaskResult:
    task: str
    total: int
    correct: int
    wrong: int
    unanswered: int
    precision: float
    recall: float
    f1: float
    model_col: str
    answer_col: Optional[str]
    correctness_basis: str


MODEL_COL_CAND = ["model", "model_name", "llm", "llm_name", "model_id", "model_used", "llm_model"]
ANSWER_COL_CAND = ["answer", "response", "model_answer", "llm_answer", "generated_answer", "output", "prediction"]
CORRECT_COL_CAND = ["is_correct", "correct", "match", "exact_match", "is_right", "right"]
SCORE_COL_CAND = ["score", "final_score", "overall_score", "accuracy_score", "rating", "grade"]

UNKNOWN_PATTERNS = [
    "i don't know",
    "i do not know",
    "unknown",
    "not found",
    "no information",
    "not in the provided context",
    "can't find",
    "cannot find",
]


def _detect_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _detect_model_col_fuzzy(df: pd.DataFrame) -> Optional[str]:
    """Fuzzy detect model column.

    Many exported CSVs use different names (e.g., 'llm_model', 'model_used').
    We first try known candidates, then fall back to any column whose name
    contains 'model' or 'llm'.
    """
    c = _detect_col(df, MODEL_COL_CAND)
    if c:
        return c
    for col in df.columns:
        name = str(col).lower()
        if "model" in name or name in {"llm"} or "llm" in name:
            return col
    return None


def _detect_answer_col_fuzzy(df: pd.DataFrame) -> Optional[str]:
    c = _detect_col(df, ANSWER_COL_CAND)
    if c:
        return c
    for col in df.columns:
        name = str(col).lower()
        if "answer" in name or "response" in name or "output" in name:
            return col
    return None


def _detect_score_col(df: pd.DataFrame) -> Optional[str]:
    c = _detect_col(df, SCORE_COL_CAND)
    if c:
        return c
    # fallback: any column containing score/rating/grade
    for c in df.columns:
        n = str(c).lower()
        if "score" in n or "rating" in n or "grade" in n:
            return c
    return None


def _is_unanswered(x) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return True
    t = str(x).strip().lower()
    if not t:
        return True
    return any(p in t for p in UNKNOWN_PATTERNS)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _compute_task(task: str, path: Path, *, threshold: float) -> TaskResult:
    df = pd.read_csv(path)

    # Support two CSV layouts:
    # 1) "long" format: each row is one model run, with a model column.
    # 2) "wide" format: each row contains multiple model outputs, e.g. mistral_answer, mistral_correctness.

    wide_answer = None
    wide_score = None
    for c in df.columns:
        if str(c).lower() == 'mistral_answer':
            wide_answer = c
        if str(c).lower() == 'mistral_correctness':
            wide_score = c

    if wide_answer is not None and wide_score is not None:
        # Wide format: treat each question as one sample for mistral.
        mdf = df.copy()
        model_col = 'wide:mistral'
        answer_col = wide_answer
        correct_col = None
        score_col = wide_score
    else:
        # Long format fallback.
        model_col = _detect_model_col_fuzzy(df)
        if not model_col:
            raise RuntimeError(
                f"No model column found in {path}. Available columns: {list(df.columns)}"
            )

        mdf = df[df[model_col].astype(str).str.lower().str.contains("mistral")].copy()
        if mdf.empty:
            raise RuntimeError(f"No mistral rows found in {path} (model column '{model_col}')")

        answer_col = _detect_answer_col_fuzzy(mdf)
        correct_col = _detect_col(mdf, CORRECT_COL_CAND)
        score_col = _detect_score_col(mdf)

    if answer_col is None:
        unanswered_mask = pd.Series([False] * len(mdf), index=mdf.index)
    else:
        unanswered_mask = mdf[answer_col].apply(_is_unanswered)

    if correct_col is not None:
        correct_mask = mdf[correct_col].astype(str).str.lower().isin(["1", "true", "yes", "correct"])
        correctness_basis = f"{correct_col} in {{1,true,yes,correct}}"
    else:
        if score_col is None:
            raise RuntimeError(
                f"No correctness column ({CORRECT_COL_CAND}) and no score column found in {path}. "
                f"Available columns: {list(mdf.columns)}"
            )
        s = pd.to_numeric(mdf[score_col], errors="coerce").fillna(0)
        correct_mask = s >= float(threshold)
        correctness_basis = f"{score_col}>={threshold}"

    correct = int((correct_mask & ~unanswered_mask).sum())
    unanswered = int(unanswered_mask.sum())
    wrong = int((~correct_mask & ~unanswered_mask).sum())
    total = int(len(mdf))

    # Metric definition:
    # - Single/Multi: positive = correct answer produced.
    # - Null: positive = abstention.
    if task in {"Single", "Multi"}:
        TP, FP, FN = correct, wrong, wrong + unanswered
    else:  # Null
        TP, FP, FN = unanswered, (correct + wrong), 0

    precision = _safe_div(TP, TP + FP)
    recall = _safe_div(TP, TP + FN)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return TaskResult(
        task=task,
        total=total,
        correct=correct,
        wrong=wrong,
        unanswered=unanswered,
        precision=precision,
        recall=recall,
        f1=f1,
        model_col=str(model_col),
        answer_col=answer_col,
        correctness_basis=correctness_basis,
    )


def _render_md(results: list[TaskResult], *, threshold: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d")

    lines: list[str] = []
    lines.append(f"# Mistral 模型在 Single / Multi / Null 三类数据集上的统计与指标（生成于 {now}）")
    lines.append("")
    lines.append("数据来源：")
    lines.append("- `compare/llm_compare_20260401_131033_scored.csv`（Single）")
    lines.append("- `compare/llm_compare_20260401_154508_Multi_scored.csv`（Multi）")
    lines.append("- `compare/llm_compare_20260401_184038_Null_scored.csv`（Null）")
    lines.append("")
    lines.append("## 1. 统计口径")
    lines.append("")
    lines.append("### 1.1 模型筛选")
    lines.append("在 CSV 中查找模型列（`model` / `model_name` / `llm` / `llm_name`），筛选值包含 `mistral`（大小写不敏感）的行。")
    lines.append("")
    lines.append("### 1.2 未回答（Unanswered）判定")
    lines.append("若答案文本为空或包含以下关键词，则记为未回答：")
    lines.append("`I don't know` / `unknown` / `not found` / `no information` / `not in the provided context` / `cannot find` 等。")
    lines.append("")
    lines.append("### 1.3 答对（Correct）判定")
    lines.append("优先使用显式正确性列：`is_correct` / `correct` / `match` / `exact_match`。")
    lines.append("若不存在，则使用评分列（`score` 或任一包含 `score` 的列），并以阈值判定：")
    lines.append(f"- `score >= {threshold}` 视为答对")
    lines.append("")
    lines.append("## 2. 指标定义（Precision / Recall / F1）")
    lines.append("")
    lines.append("- Single/Multi：TP=Correct，FP=Wrong，FN=Wrong+Unanswered")
    lines.append("- Null：正确行为是拒答（Unanswered），因此 TP=Unanswered，FP=Answered，FN=0")
    lines.append("")
    lines.append("## 3. 结果汇总（mistral）")
    lines.append("")
    lines.append("| Task | Total | Correct | Wrong | Unanswered | Precision | Recall | F1 |")
    lines.append("|------|-------|---------|-------|------------|-----------|--------|----|")
    for r in results:
        lines.append(
            f"| {r.task} | {r.total} | {r.correct} | {r.wrong} | {r.unanswered} | "
            f"{r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} |"
        )

    lines.append("")
    lines.append("## 4. 解析信息（用于复现/审计）")
    lines.append("")
    for r in results:
        lines.append(f"- **{r.task}**: model_col=`{r.model_col}`, answer_col=`{r.answer_col}`, correctness=`{r.correctness_basis}`")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate mistral metrics report from scored CSVs")
    parser.add_argument("--threshold", type=float, default=4.0, help="Score threshold treated as correct when no explicit correctness column exists")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    files = {
        "Single": base / "compare" / "llm_compare_20260401_131033_scored.csv",
        "Multi": base / "compare" / "llm_compare_20260401_154508_Multi_scored.csv",
        "Null": base / "compare" / "llm_compare_20260401_184038_Null_scored.csv",
    }

    results: list[TaskResult] = []
    for task, path in files.items():
        if not path.exists():
            raise SystemExit(f"Missing file: {path}")
        results.append(_compute_task(task, path, threshold=float(args.threshold)))

    md = _render_md(results, threshold=float(args.threshold))

    out_path = base / "compare" / f"mistral_metrics_report_{datetime.now().strftime('%Y%m%d')}.md"
    out_path.write_text(md, encoding="utf-8")
    print(str(out_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
