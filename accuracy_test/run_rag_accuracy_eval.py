"""Batch accuracy evaluation for this project's RAG system.

Goal
----
- Run 3 test sets (Multi/Single/Null) end-to-end through the existing RAG pipeline.
- Compute precision/recall/F1 per type + overall combined.
- Emit per-question sample analysis (CSV/JSON) and a thesis-ready Markdown report.

This script is *standalone* and does not modify the existing RAG system.
All outputs are written under `accuracy_test/`.

Metric definitions (lightweight + reproducible)
----------------------------------------------
We cast each question into an extraction-style evaluation:

Single:
- Gold is a single target string.
- We compute token-level precision/recall/F1 between prediction and gold.

Multi:
- Gold contains multiple items separated by '&' (occasionally ';' or ','), representing a set.
- We extract predicted items by checking whether each gold item appears as a substring in the
  prediction (case-insensitive); this avoids relying on the model to format a list.
- Per-question precision = |pred_items ∩ gold_items| / |pred_items| (if pred_items empty => 0)
- Per-question recall    = |pred_items ∩ gold_items| / |gold_items|
- Per-question F1        = harmonic mean.

Null:
- Gold is 'Insufficient information' for all questions.
- Treat "correct abstention" as the positive class.
  - predicted_positive = model abstains (IDK-like)
  - gold_positive      = always 1
  - precision = (# abstentions) / (# answers)  (here denom==N)
  - recall    = (# abstentions) / N
  - F1        = same as precision/recall (since gold positive for all)

Overall:
- Micro-average across all questions: sum TP / sum (TP+FP) etc.
  - For Single, TP/FP/FN are computed on token overlap.
  - For Multi, TP/FP/FN are computed on set overlap.
  - For Null, TP/FP/FN are computed on abstention vs non-abstention.

Outputs
-------
accuracy_test/
  runs/<timestamp>_<llm>_<collection>/
    predictions.jsonl
    per_question.csv
    per_question.json
    summary.json
    report.md

Usage
-----
python -m accuracy_test.run_rag_accuracy_eval --config config/config.yaml --key_file encryption.key

Notes
-----
- Requires the vector DB collection to be populated and Ollama available, same as `rag_system.py` CLI.
- The script runs queries sequentially; set --limit for quick smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Import the existing runtime constructor without modifying it.
from src.rag_pipeline.rag_system import _build_runtime


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


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _strip_answer_prefix(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return ""
    if t.lower().startswith("answer:"):
        t = t.split(":", 1)[1].strip()
    return t


def _is_idk(answer_text: str) -> bool:
    a = _norm(_strip_answer_prefix(answer_text))
    return any(p in a for p in IDK_PATTERNS) or a in {"idk", "i dont know"}


def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", _norm(s))


def _prf_from_counts(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _split_multi_gold(gold: str) -> List[str]:
    g = (gold or "").strip()
    if not g:
        return []

    if _norm(g) in {"yes", "no"}:
        return [_norm(g)]

    if "&" in g:
        parts = [p.strip() for p in g.split("&")]
    elif ";" in g:
        parts = [p.strip() for p in g.split(";")]
    elif "," in g:
        parts = [p.strip() for p in g.split(",")]
    else:
        parts = [p.strip() for p in re.split(r"\band\b", g, flags=re.IGNORECASE) if p.strip()]

    out: List[str] = []
    for p in parts:
        p = p.strip().strip('"\'')
        p = re.sub(r"\s+", " ", p)
        if p:
            out.append(_norm(p))
    return out or [_norm(g)]


@dataclass
class QuestionItem:
    qid: int
    question: str
    gold: str
    qtype: str  # Multi/Single/Null


def _load_lines(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return lines


def _load_dataset(*, queries_path: Path, gold_path: Path, qtype: str) -> List[QuestionItem]:
    qs = _load_lines(queries_path)
    golds = _load_lines(gold_path)
    if len(qs) != len(golds):
        raise ValueError(f"Query/gold length mismatch for {qtype}: {len(qs)} vs {len(golds)}")
    return [QuestionItem(i + 1, q, g, qtype) for i, (q, g) in enumerate(zip(qs, golds))]


def _eval_single(pred: str, gold: str) -> Dict[str, Any]:
    pred_toks = _tokenize(_strip_answer_prefix(pred))
    gold_toks = _tokenize(gold)

    pred_set = set(pred_toks)
    gold_set = set(gold_toks)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p, r, f1 = _prf_from_counts(tp, fp, fn)

    exact = 1.0 if _norm(_strip_answer_prefix(pred)) == _norm(gold) else 0.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": p,
        "recall": r,
        "f1": f1,
        "exact_match": exact,
        "gold_tokens": gold_toks,
        "pred_tokens": pred_toks,
    }


def _eval_multi(pred: str, gold: str) -> Dict[str, Any]:
    gold_items = _split_multi_gold(gold)
    pred_text = _norm(_strip_answer_prefix(pred))

    # Predict an item if it appears in the answer.
    pred_items = [it for it in gold_items if it and it in pred_text]

    gold_set = set(gold_items)
    pred_set = set(pred_items)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)  # usually 0 with this extractor
    fn = len(gold_set - pred_set)
    p, r, f1 = _prf_from_counts(tp, fp, fn)

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": p,
        "recall": r,
        "f1": f1,
        "gold_items": gold_items,
        "pred_items": pred_items,
    }


def _eval_null(pred: str) -> Dict[str, Any]:
    abstain = 1.0 if _is_idk(pred) else 0.0
    # gold is always abstain=1
    tp = abstain
    fp = 0.0 if abstain else 1.0
    fn = 0.0
    p, r, f1 = _prf_from_counts(tp, fp, fn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": f1,
        "abstain": bool(abstain),
    }


def _evaluate_item(item: QuestionItem, prediction: str) -> Dict[str, Any]:
    if item.qtype == "Single":
        m = _eval_single(prediction, item.gold)
    elif item.qtype == "Multi":
        m = _eval_multi(prediction, item.gold)
    elif item.qtype == "Null":
        m = _eval_null(prediction)
    else:
        raise ValueError(f"Unknown qtype: {item.qtype}")

    return {
        "qid": item.qid,
        "type": item.qtype,
        "question": item.question,
        "gold": item.gold,
        "prediction": prediction,
        **m,
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
            "precision": sum(r["precision"] for r in rows) / len(rows) if rows else 0.0,
            "recall": sum(r["recall"] for r in rows) / len(rows) if rows else 0.0,
            "f1": sum(r["f1"] for r in rows) / len(rows) if rows else 0.0,
        },
    }

    # optional extras
    exacts = [float(r.get("exact_match", 0.0)) for r in rows if "exact_match" in r]
    if exacts:
        out["exact_match"] = sum(exacts) / len(exacts)

    null_abstain = [1.0 for r in rows if r.get("type") == "Null" and r.get("abstain")]
    null_total = [1.0 for r in rows if r.get("type") == "Null"]
    if null_total:
        out["null_abstain_rate"] = sum(null_abstain) / len(null_total)

    return out


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    # Union all keys across rows because different question types produce different debug fields.
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


def _flush_outputs(*, out_dir: Path, pred_path: Path, per_q_path: Path, per_q_rows: List[Dict[str, Any]], pred_last: Optional[Dict[str, Any]] = None) -> None:
    """Persist intermediate artifacts so long runs are observable and partially recoverable."""
    # predictions/per_question are already appended; just write aggregate views.
    _write_csv(out_dir / "per_question.csv", per_q_rows)
    _write_json(out_dir / "per_question.json", per_q_rows)

    # lightweight checkpoint
    ckpt = {
        "num_scored": len(per_q_rows),
        "last_item": {k: pred_last.get(k) for k in ("type", "qid", "question") if pred_last} if pred_last else None,
    }
    _write_json(out_dir / "checkpoint.json", ckpt)


def _render_report(*, out_dir: Path, run_meta: Dict[str, Any], summary: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# RAG Accuracy Evaluation Report\n")
    lines.append(f"- Generated: {run_meta['timestamp']}\n")
    lines.append(f"- LLM model: {run_meta.get('llm_name','unknown')}\n")
    lines.append(f"- Collection: {run_meta.get('collection_name','(config)')}\n")
    lines.append(f"- Config: `{run_meta.get('config_path','')}`\n")

    lines.append("\n## Summary (Precision / Recall / F1)\n")
    for k in ("Multi", "Single", "Null", "Overall"):
        s = summary[k]["micro"]
        lines.append(f"- **{k}**: P={s['precision']:.4f}, R={s['recall']:.4f}, F1={s['f1']:.4f} (n={summary[k]['num_samples']})\n")

    lines.append("\n## Notes on metric definitions\n")
    lines.append("- Single: token-level overlap between prediction and gold.\n")
    lines.append("- Multi: gold list extracted by '&' etc; predicted items detected via substring match.\n")
    lines.append("- Null: correct behavior is abstention; counted as positive.\n")
    lines.append("- Overall: micro-average on TP/FP/FN summed across all questions.\n")

    lines.append("\n## Artifacts\n")
    lines.append("- `predictions.jsonl`: raw RAG outputs per question (answer + diagnostics).\n")
    lines.append("- `per_question.csv` / `per_question.json`: per-sample PRF and debug fields.\n")
    lines.append("- `summary.json`: aggregate metrics.\n")

    (out_dir / "report.md").write_text("".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch RAG accuracy evaluation (Multi/Single/Null)")
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--key_file", type=str, default="encryption.key")
    p.add_argument("--collection_name", type=str, default=None)
    p.add_argument("--allow_empty_collection", action="store_true")

    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_tokens", type=int, default=None)

    p.add_argument("--limit", type=int, default=None, help="Limit questions per dataset for quick tests")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between questions")

    p.add_argument("--out_dir", type=str, default=None, help="Override output directory")
    return p


def _prepare_isolated_qdrant_storage(*, root: Path, out_dir: Path) -> Optional[Path]:
    """Create an isolated copy of local Qdrant storage for this run.

    Why:
    - qdrant-client local mode uses an exclusive lock on `qdrant_storage/.lock`.
    - On Windows, stale locks or other processes can cause `already accessed` errors.

    Approach:
    - Copy `root/qdrant_storage` to `out_dir/qdrant_storage_copy` and remove `.lock` inside.
    - Point config.storage_path to this copy.

    Returns the new storage path, or None if the source doesn't exist.
    """
    src = root / "qdrant_storage"
    if not src.exists():
        return None

    dst = out_dir / "qdrant_storage_copy"
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)

    def _ignore_lock(dir_path: str, names: List[str]) -> List[str]:
        # Skip Qdrant local lock file if present (often locked on Windows).
        return [".lock"] if ".lock" in names else []

    shutil.copytree(src, dst, ignore=_ignore_lock)

    # Remove any lock file in the copied storage.
    try:
        lock_file = dst / ".lock"
        if lock_file.exists():
            lock_file.unlink()
    except Exception:
        pass

    return dst


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    root = Path(__file__).resolve().parents[1]

    datasets = [
        ("Multi", root / "data" / "test_datasets" / "lihua-queries1", root / "data" / "gold-answer" / "lihua-queries1-gold-answer"),
        ("Single", root / "data" / "test_datasets" / "lihua-queries2", root / "data" / "gold-answer" / "lihua-queries2-gold-answer"),
        ("Null", root / "data" / "test_datasets" / "lihua-queries3", root / "data" / "gold-answer" / "lihua-queries3-gold-answer"),
    ]

    # Determine output directory early (needed for isolated storage copy).
    llm_name_for_dir = "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_collection = (args.collection_name or "config").replace("/", "_").replace("\\", "_")
    out_dir = Path(args.out_dir) if args.out_dir else (root / "accuracy_test" / "runs" / f"{timestamp}_{llm_name_for_dir}_{safe_collection}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create isolated copy of qdrant storage for this evaluation run.
    isolated_storage = _prepare_isolated_qdrant_storage(root=root, out_dir=out_dir)

    # If we made an isolated storage copy, write a patched config next to outputs and use it.
    effective_config_path = Path(args.config)
    if isolated_storage is not None:
        import yaml

        cfg_obj = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        cfg_obj.setdefault("vector_db", {})
        cfg_obj["vector_db"]["storage_path"] = str(isolated_storage)

        patched_cfg_path = out_dir / "config_patched.yaml"
        patched_cfg_path.write_text(yaml.safe_dump(cfg_obj, sort_keys=False, allow_unicode=True), encoding="utf-8")
        effective_config_path = patched_cfg_path

    rag, audit_logger = _build_runtime(
        config_path=str(effective_config_path),
        key_file=args.key_file,
        collection_name=args.collection_name,
        allow_empty_collection=args.allow_empty_collection,
    )

    # Determine metadata for report.
    llm_name = getattr(getattr(getattr(rag, "llm", None), "client", None), "model_name", None) or getattr(rag, "llm_name", "unknown")

    run_meta = {
        "timestamp": timestamp,
        "llm_name": llm_name,
        "collection_name": args.collection_name,
        "config_path": str(effective_config_path),
        "isolated_storage_path": str(isolated_storage) if isolated_storage else None,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "limit": args.limit,
    }
    _write_json(out_dir / "run_meta.json", run_meta)

    # Start fresh
    predictions_path = out_dir / "predictions.jsonl"
    scored_path = out_dir / "per_question.jsonl"
    predictions_path.write_text("", encoding="utf-8")
    scored_path.write_text("", encoding="utf-8")

    per_q_rows: List[Dict[str, Any]] = []

    total_planned = 0
    for qtype, q_path, g_path in datasets:
        n = len(_load_lines(q_path))
        if args.limit is not None:
            n = min(n, max(0, int(args.limit)))
        total_planned += n

    done = 0
    last_flush = 0

    try:
        for qtype, q_path, g_path in datasets:
            items = _load_dataset(queries_path=q_path, gold_path=g_path, qtype=qtype)
            if args.limit is not None:
                items = items[: max(0, int(args.limit))]

            for item in items:
                done += 1
                # Small progress log (useful for long runs)
                if done == 1 or done % 5 == 0:
                    print(f"[progress] {done}/{total_planned} ({qtype} qid={item.qid})")

                res = rag.answer_question(question=item.question, top_k=args.top_k, temperature=args.temperature, max_tokens=args.max_tokens)
                pred_row = {"type": qtype, "qid": item.qid, "question": item.question, "gold": item.gold, **res}
                _append_jsonl(predictions_path, pred_row)

                row = _evaluate_item(item, res.get("answer", ""))
                # Attach a few useful diagnostics for sample analysis.
                row["confidence"] = float(res.get("confidence", 0.0) or 0.0)
                row["weak_answer"] = bool(res.get("weak_answer", False))
                row["retrieval_empty"] = bool(res.get("retrieval_empty", False))
                row["num_chunks_retrieved"] = int(res.get("num_chunks_retrieved", 0) or 0)
                row["retrieval_time"] = float(res.get("retrieval_time", 0.0) or 0.0)
                row["generation_time"] = float(res.get("generation_time", 0.0) or 0.0)

                per_q_rows.append(row)
                _append_jsonl(scored_path, row)

                if args.sleep:
                    time.sleep(float(args.sleep))

                # Flush every 10 items so we don't lose everything on interruption.
                if len(per_q_rows) - last_flush >= 10:
                    _flush_outputs(out_dir=out_dir, pred_path=predictions_path, per_q_path=scored_path, per_q_rows=per_q_rows, pred_last=pred_row)
                    last_flush = len(per_q_rows)

    finally:
        # Always write whatever we have.
        _flush_outputs(out_dir=out_dir, pred_path=predictions_path, per_q_path=scored_path, per_q_rows=per_q_rows)

    # Summaries
    summary: Dict[str, Any] = {}
    for t in ("Multi", "Single", "Null"):
        summary[t] = _aggregate([r for r in per_q_rows if r.get("type") == t])
    summary["Overall"] = _aggregate(per_q_rows)
    summary["run_meta"] = run_meta

    _write_json(out_dir / "summary.json", summary)
    _render_report(out_dir=out_dir, run_meta=run_meta, summary=summary)

    # Basic console summary
    print("\n=== RAG Accuracy Summary ===")
    for k in ("Multi", "Single", "Null", "Overall"):
        s = summary[k]["micro"]
        print(f"{k}: P={s['precision']:.4f} R={s['recall']:.4f} F1={s['f1']:.4f} (n={summary[k]['num_samples']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
