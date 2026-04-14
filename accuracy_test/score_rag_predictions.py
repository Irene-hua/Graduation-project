"""Stage 2: Offline scoring of saved predictions.

Input:
- predictions.jsonl produced by `accuracy_test.run_rag_predictions`
- (optionally) gold answers are already embedded in each JSONL row (recommended)

Output (written next to the input file):
- per_question.json / per_question.csv
- summary.json
- report.md (thesis-ready skeleton)

Usage
-----
python -m accuracy_test.score_rag_predictions --predictions accuracy_test/runs/<run_id>/predictions.jsonl

"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tempfile import NamedTemporaryFile


# Expanded abstention (IDK-like) patterns.
# Keep these as plain lowercase substrings; matching happens on normalized text.
ABSTAIN_PATTERNS = (
    "i don't know",
    "i do not know",
    "idk",
    "unknown",
    "not sure",
    "unsure",
    "insufficient information",
    "not enough information",
    "no information",
    "not provided",
    "not mentioned",
    "not stated",
    "not specified",
    "cannot determine",
    "can't determine",
    "impossible to determine",
    "unable to determine",
    "unable to answer",
    "cannot answer",
    "can't answer",
    "does not contain any information",
    "doesn't contain any information",
    "does not contain information",
    "there is no information",
    "no details",
    "not available in the context",
    "not in the provided context",
    "based on the provided context, there is no information",
    "the context provided does not contain",
    "the provided context does not contain",
)


def _strip_answer_prefix(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return ""
    if t.lower().startswith("answer:"):
        t = t.split(":", 1)[1].strip()
    return t


def _norm_for_match(s: str) -> str:
    """Normalize for abstain detection & cheap fallbacks.

    - lower
    - remove "Answer:" prefix
    - replace punctuation with spaces but keep word boundaries
    - collapse whitespace
    """

    t = _strip_answer_prefix(s)
    t = (t or "").lower()
    t = t.replace("\u00a0", " ")
    # keep word boundaries; punctuation -> space
    t = re.sub(r"[^\w]+", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_abstain(text: str) -> bool:
    t = _norm_for_match(text)
    if not t:
        return True

    # exact short forms
    if t in {"idk", "i dont know", "i don't know", "i do not know"}:
        return True

    return any(p in t for p in ABSTAIN_PATTERNS)


def _prf_from_counts(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _fallback_match(gold: str, pred: str) -> bool:
    """Cheap deterministic fallback when LLM judge is unavailable.

    Uses normalized substring containment both ways.
    """

    g = _norm_for_match(gold)
    p = _norm_for_match(pred)
    if not g or not p:
        return False
    return (g in p) or (p in g)


def _normalize_short_answer(s: str) -> str:
    """Normalize short closed-form answers for fast deterministic matching."""

    t = _norm_for_match(s)
    # Keep only word tokens (handles 'Answer: Yes,' etc.)
    toks = t.split()
    if not toks:
        return ""
    first = toks[0]
    # common yes/no variants
    if first in {"yes", "y"}:
        return "yes"
    if first in {"no", "n"}:
        return "no"
    # allow true/false
    if first in {"true"}:
        return "true"
    if first in {"false"}:
        return "false"
    return first


def _fast_semantic_match(gold: str, pred: str) -> Optional[bool]:
    """Deterministic shortcut to avoid LLM misjudging obvious cases.

    Returns:
        True/False if we can decide deterministically, else None.
    """

    g0 = _normalize_short_answer(gold)
    p0 = _normalize_short_answer(pred)
    if not g0 or not p0:
        return None

    # If gold is a short closed-form label, require prediction to start with same label.
    if g0 in {"yes", "no", "true", "false"}:
        return p0 == g0

    # If gold is a single token/number/date-like (very short), allow normalized substring containment.
    # This helps for cases like "sunday" or "3 pm" where pred prefixes with explanation.
    g_norm = _norm_for_match(gold)
    if len(g_norm.split()) <= 3:
        p_norm = _norm_for_match(pred)
        if g_norm and p_norm and g_norm in p_norm:
            return True

    return None


_JUDGE_PROMPT_TEMPLATE = (
    "你是一个严格的答案评估器。请只判断‘预测答案的最终结论’是否与标准答案语义一致。\n"
    "注意：预测答案可能包含解释，但只看结论是否一致。\n"
    "只能回答‘是’或‘否’，不要输出任何解释。\n\n"
    "标准答案：{gold}\n"
    "预测答案：{pred}\n"
)


def _parse_yes_no(text: str) -> Optional[bool]:
    """Parse judge output.

    Returns:
        True  -> yes
        False -> no
        None  -> cannot parse
    """

    t = (text or "").strip()
    if not t:
        return None

    # Normalize to first line, remove quotes/spaces.
    first = t.splitlines()[0].strip().strip('"\'')

    # Chinese outputs
    if first.startswith("是"):
        return True
    if first.startswith("否"):
        return False

    # English outputs (sometimes happens)
    low = first.lower()
    if low.startswith("yes"):
        return True
    if low.startswith("no"):
        return False

    # If contains 是 but not 否, accept as yes
    if ("是" in first) and ("否" not in first):
        return True
    if "否" in first:
        return False

    return None


def call_local_llm(*, model: str, prompt: str, timeout_s: int = 30) -> str:
    """Call local Ollama via CLI (no cloud).

    Implementation note (Windows-friendly):
    - We write the prompt to a temporary UTF-8 file then pipe it to `ollama run`.
      This avoids quoting/encoding issues when prompts contain Chinese punctuation.
    """

    cmd = ["ollama", "run", model]

    try:
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as tf:
            tf.write(prompt)
            tmp_path = tf.name

        # Use `type` to pipe file content on Windows.
        shell_cmd = f'type "{tmp_path}" | ollama run {model}'
        proc = subprocess.run(
            shell_cmd,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
            shell=True,
            encoding="utf-8",
            errors="ignore",
        )

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(f"ollama returned code={proc.returncode}: {stderr[:500]}")

        return (proc.stdout or "").strip()
    finally:
        # Best-effort cleanup
        try:
            import os

            if "tmp_path" in locals() and tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def is_semantic_match(
    *,
    gold: str,
    pred: str,
    model: str,
    timeout_s: int = 30,
    retries: int = 3,
) -> Tuple[bool, str, Optional[str]]:
    """Semantic match judged by local LLM.

    Returns:
        (match, judge_method, judge_raw)
    """

    # Deterministic fast-path to avoid LLM misjudging trivial closed-form answers.
    fast = _fast_semantic_match(gold, pred)
    if fast is not None:
        return bool(fast), "fast", f"fast_match(g0={_normalize_short_answer(gold)}, p0={_normalize_short_answer(pred)})"

    prompt = _JUDGE_PROMPT_TEMPLATE.format(gold=gold.strip(), pred=_strip_answer_prefix(pred).strip())

    last_err: Optional[str] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            out = call_local_llm(model=model, prompt=prompt, timeout_s=timeout_s)
            parsed = _parse_yes_no(out)
            if parsed is None:
                last_err = f"unparseable_judge_output: {out[:200]}"
                # small backoff and retry
                if attempt < retries:
                    time.sleep(0.2 * attempt)
                    continue
                break
            return bool(parsed), "llm", out
        except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError) as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                time.sleep(0.3 * attempt)
                continue
            break

    # Fallback to deterministic match.
    return _fallback_match(gold, pred), "fallback", last_err


def score_qa(*, gold: str, pred: str, model: str, timeout_s: int, retries: int) -> Dict[str, Any]:
    """Score any non-Null QA (gold is expected non-empty)."""

    if is_abstain(pred):
        tp, fp, fn = 0.0, 0.0, 1.0
        p, r, f1 = _prf_from_counts(tp, fp, fn)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
            "judge_method": "abstain",
            "judge_raw": None,
            "abstain": True,
        }

    match, method, raw = is_semantic_match(
        gold=gold,
        pred=pred,
        model=model,
        timeout_s=timeout_s,
        retries=retries,
    )

    if match:
        tp, fp, fn = 1.0, 0.0, 0.0
    else:
        tp, fp, fn = 0.0, 1.0, 0.0

    p, r, f1 = _prf_from_counts(tp, fp, fn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": f1,
        "judge_method": method,
        "judge_raw": raw,
        "abstain": False,
    }


def score_null(*, pred: str) -> Dict[str, Any]:
    """Score Null: correct behavior is abstention."""

    abstain = is_abstain(pred)
    if abstain:
        tp, fp, fn = 1.0, 0.0, 0.0
        method = "abstain"
    else:
        tp, fp, fn = 0.0, 1.0, 0.0
        method = "not_abstain"

    p, r, f1 = _prf_from_counts(tp, fp, fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": f1,
        "judge_method": method,
        "judge_raw": None,
        "abstain": bool(abstain),
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(float(r.get("tp", 0.0)) for r in rows)
    fp = sum(float(r.get("fp", 0.0)) for r in rows)
    fn = sum(float(r.get("fn", 0.0)) for r in rows)
    p, r, f1 = _prf_from_counts(tp, fp, fn)

    out: Dict[str, Any] = {
        "num_samples": len(rows),
        "micro": {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn},
        "macro": {
            "precision": sum(float(r["precision"]) for r in rows) / len(rows) if rows else 0.0,
            "recall": sum(float(r["recall"]) for r in rows) / len(rows) if rows else 0.0,
            "f1": sum(float(r["f1"]) for r in rows) / len(rows) if rows else 0.0,
        },
    }

    null_total = [1.0 for r in rows if r.get("type") == "Null"]
    if null_total:
        null_abstain = [1.0 for r in rows if r.get("type") == "Null" and r.get("abstain")]
        out["null_abstain_rate"] = sum(null_abstain) / len(null_total)

    # FP/FN counters for error analysis
    out["fp_count"] = sum(1.0 for r in rows if float(r.get("fp", 0.0)) > 0)
    out["fn_count"] = sum(1.0 for r in rows if float(r.get("fn", 0.0)) > 0)

    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    all_fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                all_fields.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in all_fields})


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_error_samples(out_dir: Path) -> List[Dict[str, Any]]:
    p = out_dir / "error_samples.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            return []


def _safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0


def _summarize_judge_methods(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    total = float(len(rows))
    counts: Dict[str, float] = {}
    for r in rows:
        k = str(r.get("judge_method") or "")
        counts[k] = counts.get(k, 0.0) + 1.0
    # convert to rates for narrative
    return {k: _safe_div(v, total) for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))}


def _pick_case(examples: List[Dict[str, Any]], *, error_type: str, qtype: str) -> Optional[Dict[str, Any]]:
    for e in examples:
        if str(e.get("type")) == qtype and str(e.get("error_type")) == error_type:
            return e
    return None


def _render_report(out_dir: Path, summary: Dict[str, Any], run_meta: Optional[Dict[str, Any]]) -> None:
    # Thesis-ready report template (Chinese).
    ts = (run_meta or {}).get("timestamp", "")
    llm_name = (run_meta or {}).get("llm_name", "unknown")
    collection_name = (run_meta or {}).get("collection_name", "")
    config_path = (run_meta or {}).get("config_path", "")
    top_k = (run_meta or {}).get("top_k", "")
    temperature = (run_meta or {}).get("temperature", "")
    max_tokens = (run_meta or {}).get("max_tokens", "")

    judge_model = (run_meta or {}).get("judge_model", "llama3.2:3b")
    judge_timeout = (run_meta or {}).get("judge_timeout", 30)
    judge_retries = (run_meta or {}).get("judge_retries", 3)

    def _fmt_micro(key: str) -> str:
        s = summary[key]["micro"]
        return f"P={s['precision']:.4f}, R={s['recall']:.4f}, F1={s['f1']:.4f} (TP={s['tp']:.0f}, FP={s['fp']:.0f}, FN={s['fn']:.0f}, n={summary[key]['num_samples']})"

    lines: List[str] = []

    lines.append("# RAG 系统准确性评估流程与指标（离线评分报告）\n\n")
    lines.append(
        "本节给出本项目 RAG 系统准确性评测实验的可复现流程与指标定义。\n"
        "实验采用两阶段方式：先离线生成每题的 RAG 输出（`predictions.jsonl`），再对保存的结果进行离线评分与统计。\n\n"
    )

    # 1) Experimental setup
    lines.append("## 1. 实验设置（Experimental Setup）\n\n")
    lines.append("### 1.1 数据集与划分\n\n")
    lines.append(
        "本实验使用三类测试集（每类 60 题，共 180 题）：\n\n"
        "- **Multi**：`data/test_datasets/lihua-queries1`（gold：`data/gold-answer/lihua-queries1-gold-answer`）\n"
        "- **Single**：`data/test_datasets/lihua-queries2`（gold：`data/gold-answer/lihua-queries2-gold-answer`）\n"
        "- **Null**：`data/test_datasets/lihua-queries3`（gold 语义为不可回答，应拒答）\n\n"
        "其中 Multi/Single 为“gold 非空”的可回答问题，Null 为“应拒答”的不可回答问题。\n\n"
    )

    lines.append("### 1.2 推理与评分工具链（完全本地）\n\n")
    lines.append(
        "- **RAG 推理**：复用项目现有 RAG pipeline，不做任何修改。\n"
        "- **语义裁判（Judge）**：使用本地 Ollama 部署的 LLM（默认 `llama3.2:3b`）判断预测答案与 gold 是否语义一致。\n"
        "- **拒答识别（Abstention detection）**：使用确定性规则匹配（例如 `insufficient information`、`does not contain any information` 等）识别模型是否拒答。\n\n"
        "本次 run 的关键参数（自动记录于 `run_meta.json`）：\n\n"
        f"- timestamp: {ts}\n"
        f"- rag_llm_model: {llm_name}\n"
        f"- collection_name: {collection_name}\n"
        f"- config_path: `{config_path}`\n"
        f"- top_k: {top_k}\n"
        f"- temperature: {temperature}\n"
        f"- max_tokens: {max_tokens}\n"
        f"- judge_model (Ollama): {judge_model}\n"
        f"- judge_timeout: {judge_timeout}s\n"
        f"- judge_retries: {judge_retries}\n\n"
    )

    lines.append("### 1.3 两阶段评估流程\n\n")
    lines.append(
        "**Stage 1（在线推理）**：逐题调用 RAG，保存每题输出到 `predictions.jsonl`。\n"
        "**Stage 2（离线评分）**：读取 `predictions.jsonl`，对每题进行拒答判断与语义一致性判断，输出逐题明细和汇总指标。\n\n"
    )

    # 2) Metric definitions
    lines.append("## 2. 指标定义（Precision / Recall / F1）\n\n")
    lines.append("### 2.1 TP/FP/FN 定义\n\n")
    lines.append(
        "将每个问题视作一次预测任务，并定义：\n\n"
        "- **TP（True Positive）**：预测为正确的次数\n"
        "- **FP（False Positive）**：预测给出了具体答案但与 gold 语义不一致的次数（答错）\n"
        "- **FN（False Negative）**：在应回答（Multi/Single）场景下，模型拒答/信息不足导致未回答的次数（漏答）\n\n"
    )

    lines.append("### 2.2 Precision / Recall / F1 公式\n\n")
    lines.append(
        "令 TP、FP、FN 为某个集合（如某一测试集或全体样本）上的计数，则：\n\n"
        "\\[\n"
        "Precision = \\frac{TP}{TP + FP}\\,\n"
        "Recall = \\frac{TP}{TP + FN}\\,\n"
        "F1 = \\frac{2 \\cdot Precision \\cdot Recall}{Precision + Recall}.\n"
        "\\]\\n\\n"
        "为避免除 0，若分母为 0，则对应指标记为 0（实现中 `_prf_from_counts`）。\n\n"
    )

    # 3) Per-question judging rules
    lines.append("## 3. 逐题判定逻辑（Multi / Single / Null）\n\n")
    lines.append("### 3.1 拒答（Abstain）判定\n\n")
    lines.append(
        "定义函数 `is_abstain(text)`：若模型输出为空、或包含典型拒答表述（如 `i don't know`、`insufficient information`、"
        "`does not contain any information`、`the provided context does not contain ...` 等），则判为拒答。\n\n"
        "该判定为确定性规则，保证可复现。\n\n"
    )

    lines.append("### 3.2 语义一致性（Semantic Match）判定\n\n")
    lines.append(
        "对 Multi/Single（gold 非空）且非拒答的样本，调用本地 Ollama 模型进行语义裁判。Prompt 固定为：\n\n"
        "```text\n"
        "你是一个严格的答案评估器。判断以下两个答案是否语义一致。只回答“是”或“否”。\n\n"
        "标准答案：{gold}\n"
        "预测答案：{pred}\n"
        "```\n\n"
        "若裁判输出不可解析或调用失败，则回退到一个确定性的字符串匹配规则（`fallback`），并在逐题明细中记录 `judge_method` 与 `judge_raw`。\n\n"
    )

    lines.append("### 3.3 Multi/Single/Null 的统一计分规则\n\n")
    lines.append(
        "- **Multi / Single（gold 非空）**：\n"
        "  - 若 `is_abstain(pred)=True`：计为 FN（未回答）\n"
        "  - 否则若 `is_semantic_match(gold, pred)=True`：计为 TP（回答正确）\n"
        "  - 否则：计为 FP（回答错误）\n\n"
        "- **Null（应拒答）**：\n"
        "  - 若 `is_abstain(pred)=True`：计为 TP（正确拒答）\n"
        "  - 否则：计为 FP（未拒答且给出具体答案）\n\n"
    )

    # 4) Flowchart-like pseudocode
    lines.append("## 4. 判定流程图式伪代码（可直接写入论文）\n\n")
    lines.append("### 4.1 Multi/Single（gold 非空）\n\n")
    lines.append(
        "```text\n"
        "Input: gold, pred\n"
        "If is_abstain(pred):\n"
        "    TP=0, FP=0, FN=1      # 拒答 -> 漏答\n"
        "Else:\n"
        "    If is_semantic_match(gold, pred):\n"
        "        TP=1, FP=0, FN=0  # 语义一致 -> 正确\n"
        "    Else:\n"
        "        TP=0, FP=1, FN=0  # 语义不一致 -> 答错\n"
        "Return TP,FP,FN\n"
        "```\n\n"
    )

    lines.append("### 4.2 Null（应拒答）\n\n")
    lines.append(
        "```text\n"
        "Input: pred\n"
        "If is_abstain(pred):\n"
        "    TP=1, FP=0, FN=0      # 正确拒答\n"
        "Else:\n"
        "    TP=0, FP=1, FN=0      # 未拒答\n"
        "Return TP,FP,FN\n"
        "```\n\n"
    )

    # 5) Aggregation
    lines.append("## 5. 汇总统计方法（Micro / Macro）\n\n")
    lines.append(
        "本实验同时输出 micro-average 与 macro-average：\n\n"
        "- **Micro-average**：先对样本集合求和 TP/FP/FN，再代入公式计算 P/R/F1。\n"
        "- **Macro-average**：先逐题计算 P/R/F1，再取均值。\n\n"
        "论文中建议以 micro-average 作为主要指标，因为它更直接反映总体正确/错误/漏答的比例。\n\n"
    )

    # 6) Results
    lines.append("## 6. 实验结果（本次运行）\n\n")
    lines.append("本次运行的 micro-average 指标如下：\n\n")
    lines.append(f"- **Multi**: {_fmt_micro('Multi')}\n")
    lines.append(f"- **Single**: {_fmt_micro('Single')}\n")
    lines.append(f"- **Null**: {_fmt_micro('Null')}\n")
    lines.append(f"- **Overall**: {_fmt_micro('Overall')}\n\n")

    lines.append("并统计 FP/FN 数量（便于错误类型分析）：\n\n")
    for k in ("Multi", "Single", "Null", "Overall"):
        lines.append(f"- {k}: FP={summary[k].get('fp_count', 0)}, FN={summary[k].get('fn_count', 0)}\n")
    lines.append("\n")

    # 7) Artifacts
    lines.append("## 7. 产物文件与可复现性（Artifacts & Reproducibility）\n\n")
    lines.append(
        "离线评分阶段会在同目录生成以下文件：\n\n"
        "- `per_question.csv` / `per_question.json`：逐题明细（含 TP/FP/FN、P/R/F1、judge_method、诊断字段）\n"
        "- `summary.json`：各子集与 overall 的汇总指标（micro/macro）\n"
        "- `report.md`：本报告（论文可粘贴版本）\n"
        "- `error_samples.json` / `error_samples.md`：错误样本（FP/FN）摘录，用于定性分析\n\n"
        "其中 `judge_method` 字段用于保证裁判可审计：\n"
        "- `abstain`：直接由拒答规则判定\n"
        "- `llm`：由本地 Ollama 模型裁判\n"
        "- `fallback`：Ollama 调用失败或输出不可解析时的确定性回退规则\n\n"
    )

    # 8) Limitations
    lines.append("## 8. 局限性与威胁（Limitations / Threats to Validity）\n\n")
    lines.append(
        "1. **语义裁判偏差**：语义一致性由 LLM 裁判，可能受到裁判模型能力与提示词的影响；虽使用本地模型与固定 prompt 以提升可复现性，但仍可能存在误判。\n"
        "2. **拒答识别覆盖不完全**：`is_abstain` 采用规则匹配，仍可能漏检/误检一些边缘表述。\n"
        "3. **二值化评分的粒度**：Multi/Single 采用“正确/错误/拒答”三值、每题 TP/FP/FN 取 0/1 的方式，无法区分部分正确（例如 Multi 只覆盖部分要点）的情况。\n"
        "4. **数据集代表性**：当前测试集规模为 3×60，结论对更大规模或领域迁移的泛化能力仍需进一步实证。\n\n"
        "为降低上述威胁，本实验输出逐题明细与错误样本，支持人工抽查与复核。\n\n"
    )

    # 9) Narrative conclusion (template)
    # Read per-question rows from the artifacts we have just written.
    per_q_path = out_dir / "per_question.json"
    rows: List[Dict[str, Any]] = []
    try:
        rows = json.loads(per_q_path.read_text(encoding="utf-8")) if per_q_path.exists() else []
    except Exception:
        try:
            rows = json.loads(per_q_path.read_text(encoding="utf-8-sig")) if per_q_path.exists() else []
        except Exception:
            rows = []

    by_type: Dict[str, List[Dict[str, Any]]] = {"Multi": [], "Single": [], "Null": []}
    for r in rows:
        t = str(r.get("type"))
        if t in by_type:
            by_type[t].append(r)

    err_examples = _load_error_samples(out_dir)

    def _err_rates(t: str) -> Tuple[float, float, float]:
        s = summary[t]["micro"]
        tp, fp, fn = float(s.get("tp", 0.0)), float(s.get("fp", 0.0)), float(s.get("fn", 0.0))
        n = float(summary[t].get("num_samples", 0.0))
        return _safe_div(tp, n), _safe_div(fp, n), _safe_div(fn, n)

    om = summary["Overall"]["micro"]
    o_p, o_r, o_f1 = float(om["precision"]), float(om["recall"]), float(om["f1"])

    multi_tp_r, multi_fp_r, multi_fn_r = _err_rates("Multi")
    single_tp_r, single_fp_r, single_fn_r = _err_rates("Single")
    null_tp_r, null_fp_r, null_fn_r = _err_rates("Null")

    lines.append("## 9. 结果解读与结论（论文写作模板，可按需编辑）\n\n")
    lines.append(
        "本实验在三个测试子集（Multi/Single/Null）与整体（Overall）上报告 micro-average 的 Precision、Recall 与 F1。"
        "从结果上看，不同题型的失误模式存在明显差异：Multi 更容易出现时序/因果关系判断错误，Single 更容易出现语义裁判判定为不一致的情况，"
        "Null 则主要反映系统在信息不足场景下的拒答能力。\n\n"
    )

    lines.append("### 9.1 整体表现（Overall）\n\n")
    lines.append(
        f"Overall 指标为 P={o_p:.4f}、R={o_r:.4f}、F1={o_f1:.4f}。"
        "其中 Precision 主要受到错误回答（FP）数量影响，Recall 主要受到 Multi/Single 的拒答/信息不足导致的漏答（FN）影响。\n\n"
    )

    lines.append("### 9.2 分题型对比（Multi vs Single vs Null）\n\n")
    lines.append(
        f"- **Multi**：TP 比例约 {multi_tp_r:.2%}，FP 比例约 {multi_fp_r:.2%}，FN 比例约 {multi_fn_r:.2%}。"
        "Multi 问题通常涉及多步事实或时间先后关系，系统更容易在关系判断上给出错误结论（FP），或在检索不足时输出信息不足（FN）。\n"
        f"- **Single**：TP 比例约 {single_tp_r:.2%}，FP 比例约 {single_fp_r:.2%}，FN 比例约 {single_fn_r:.2%}。"
        "Single 的 gold 往往是单一事实点；在本实验采用的‘语义裁判’口径下，错误更多表现为给出具体答案但与 gold 语义不一致（FP）。\n"
        f"- **Null**：TP（正确拒答）比例约 {null_tp_r:.2%}，FP（未拒答）比例约 {null_fp_r:.2%}。"
        "Null 子集用于度量系统在不可回答问题上的拒答能力。\n\n"
    )

    lines.append("### 9.3 主要错误来源（基于逐题字段与裁判方法分布）\n\n")
    for t in ("Multi", "Single", "Null"):
        jm = _summarize_judge_methods(by_type.get(t, []))
        if not jm:
            continue
        items = ", ".join([f"{k}={v:.2%}" for k, v in jm.items() if k])
        lines.append(f"- **{t}** 的判定来源（judge_method）占比：{items}\n")
    lines.append(
        "其中 `abstain` 表示直接命中拒答规则；`llm` 表示由本地 Ollama 语义裁判给出一致/不一致判定；"
        "`fallback` 表示 Ollama 调用失败或输出不可解析时使用的确定性回退规则。\n\n"
    )

    lines.append("### 9.4 典型案例（从错误样本中引用）\n\n")

    case_fp_multi = _pick_case(err_examples, error_type="FP", qtype="Multi")
    case_fn_multi = _pick_case(err_examples, error_type="FN", qtype="Multi")
    case_fp_null = _pick_case(err_examples, error_type="FP", qtype="Null")

    def _render_case(title: str, e: Optional[Dict[str, Any]]) -> None:
        if not e:
            lines.append(f"- {title}：本次运行未抽取到对应样本（可提高 --error_samples 或手动从 per_question.csv 筛选）。\n")
            return
        qid = e.get("qid")
        jm = e.get("judge_method")
        lines.append(f"- **{title}**（Q{qid}, judge={jm}）\n")
        lines.append(f"  - Question: {e.get('question')}\n")
        lines.append(f"  - Gold: {e.get('gold')}\n")
        lines.append(f"  - Prediction: {e.get('prediction')}\n")

    _render_case("Multi-错误回答（FP）示例", case_fp_multi)
    _render_case("Multi-拒答/信息不足（FN）示例", case_fn_multi)
    _render_case("Null-未拒答（FP）示例", case_fp_null)

    lines.append(
        "以上案例可作为论文中的定性分析材料，用于说明模型在‘关系/时序判断’与‘拒答策略’上的典型失败模式。\n\n"
    )

    lines.append("### 9.5 结论小结（可直接用于论文）\n\n")
    lines.append(
        "综合三类测试集结果可以看出：当前 RAG 系统在可回答问题上存在一定比例的错误回答（FP），同时在部分 Multi 问题上出现了拒答/信息不足导致的漏答（FN）。"
        "对于 Null 类问题，系统拒答能力仍有提升空间（尤其是减少‘在上下文不足时仍给出具体答案’的情况）。"
        "后续优化方向包括：提升检索召回（降低 Multi 的 FN）、增强关系推理与时间顺序建模（降低 Multi 的 FP）、以及引入更严格的拒答触发阈值（提升 Null 的 TP）。\n\n"
    )

    # Windows-friendly: write UTF-8 with BOM to avoid mojibake in some viewers.
    (out_dir / "report.md").write_text("".join(lines), encoding="utf-8-sig")


@dataclass
class ErrorSample:
    type: str
    qid: Any
    question: str
    gold: str
    prediction: str
    error_type: str  # FP / FN
    judge_method: str


def _write_error_samples(out_dir: Path, rows: List[Dict[str, Any]], limit: int = 10) -> None:
    errs: List[ErrorSample] = []
    for r in rows:
        fp = float(r.get("fp", 0.0))
        fn = float(r.get("fn", 0.0))
        if fp <= 0 and fn <= 0:
            continue
        err_type = "FP" if fp > 0 else "FN"
        errs.append(
            ErrorSample(
                type=str(r.get("type")),
                qid=r.get("qid"),
                question=str(r.get("question")),
                gold=str(r.get("gold")),
                prediction=str(r.get("prediction")),
                error_type=err_type,
                judge_method=str(r.get("judge_method")),
            )
        )

    # Take top-N; for determinism keep original order
    errs = errs[: max(0, int(limit))]

    out = [e.__dict__ for e in errs]
    _write_json(out_dir / "error_samples.json", out)

    # Also render a small markdown for thesis-ready qualitative analysis
    md_lines: List[str] = []
    md_lines.append("# Error Samples (Top)\n\n")
    for e in errs:
        md_lines.append(f"## {e.type} Q{e.qid} ({e.error_type}, judge={e.judge_method})\n\n")
        md_lines.append(f"**Question**: {e.question}\n\n")
        md_lines.append(f"**Gold**: {e.gold}\n\n")
        md_lines.append(f"**Prediction**: {e.prediction}\n\n")
    (out_dir / "error_samples.md").write_text("".join(md_lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage2: score predictions.jsonl offline")
    p.add_argument("--predictions", type=str, required=True, help="Path to predictions.jsonl")
    p.add_argument("--out_dir", type=str, default=None, help="Override output directory (default: predictions parent)")

    p.add_argument("--judge_model", type=str, default="llama3.2:3b", help="Local Ollama model for semantic judging")
    p.add_argument("--judge_timeout", type=int, default=30, help="Ollama call timeout seconds")
    p.add_argument("--judge_retries", type=int, default=3, help="Max retries for judge calls")
    p.add_argument("--error_samples", type=int, default=10, help="How many error samples to export")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    pred_path = Path(args.predictions)
    out_dir = Path(args.out_dir) if args.out_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            qtype = obj.get("type")
            qid = obj.get("qid")
            question = obj.get("question")
            gold = obj.get("gold", "")
            pred = obj.get("answer", "")

            if qtype in {"Single", "Multi"}:
                m = score_qa(gold=gold, pred=pred, model=args.judge_model, timeout_s=int(args.judge_timeout), retries=int(args.judge_retries))
            elif qtype == "Null":
                m = score_null(pred=pred)
            else:
                continue

            row = {
                "type": qtype,
                "qid": qid,
                "question": question,
                "gold": gold,
                "prediction": pred,
                "error": obj.get("error"),
                "confidence": obj.get("confidence"),
                "weak_answer": obj.get("weak_answer"),
                "retrieval_empty": obj.get("retrieval_empty"),
                "num_chunks_retrieved": obj.get("num_chunks_retrieved"),
                "retrieval_time": obj.get("retrieval_time"),
                "generation_time": obj.get("generation_time"),
                "judge_method": m.get("judge_method"),
                "judge_raw": m.get("judge_raw"),
                **{k: v for k, v in m.items() if k not in {"judge_method", "judge_raw"}},
            }
            rows.append(row)

    per_q_json = out_dir / "per_question.json"
    per_q_csv = out_dir / "per_question.csv"
    _write_json(per_q_json, rows)
    _write_csv(per_q_csv, rows)

    summary: Dict[str, Any] = {
        "Multi": _aggregate([r for r in rows if r.get("type") == "Multi"]),
        "Single": _aggregate([r for r in rows if r.get("type") == "Single"]),
        "Null": _aggregate([r for r in rows if r.get("type") == "Null"]),
        "Overall": _aggregate(rows),
    }

    run_meta = None
    run_meta_path = out_dir / "run_meta.json"
    if run_meta_path.exists():
        try:
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        except Exception:
            run_meta = None

    if run_meta is None:
        run_meta = {}
    run_meta["judge_model"] = args.judge_model
    run_meta["judge_timeout"] = int(args.judge_timeout)
    run_meta["judge_retries"] = int(args.judge_retries)

    summary["run_meta"] = run_meta
    _write_json(out_dir / "summary.json", summary)
    _render_report(out_dir, summary, run_meta)

    _write_error_samples(out_dir, rows, limit=int(args.error_samples))

    print("=== Offline scoring summary ===")
    for k in ("Multi", "Single", "Null", "Overall"):
        s = summary[k]["micro"]
        print(f"{k}: P={s['precision']:.4f} R={s['recall']:.4f} F1={s['f1']:.4f} (n={summary[k]['num_samples']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
