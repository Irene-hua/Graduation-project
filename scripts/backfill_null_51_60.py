"""Backfill missing Null qid=51..60 rows in a scored run.

This script:
- Reads <run_dir>/predictions.jsonl
- Reads <run_dir>/per_question.json
- Ensures Null qid 51..60 exist in per_question.json; if missing, it creates rows using
  the same basic scoring logic used for Null questions: abstain -> TP=1 else FP=1.
- Writes updated per_question.json and then calls accuracy_test.manual_patch_run to
  recompute summary/report/verify/computed/csv/error_samples.

Usage:
  python scripts/backfill_null_51_60.py --run_dir accuracy_test/runs/<run_id>

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


ABSTAIN_SUBSTRINGS = [
    "i don't know",
    "i do not know",
    "insufficient information",
    "not mentioned",
    "no information",
    "cannot determine",
    "does not provide",
    "context does not",
    "not provide",
]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def _is_abstain(text: str | None) -> bool:
    if not text:
        return True
    t = text.strip().lower()
    if t.startswith("answer:"):
        t = t[len("answer:") :].strip()
    if not t:
        return True
    return any(s in t for s in ABSTAIN_SUBSTRINGS)


def _prf(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _read_predictions(pred_path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with pred_path.open("rb") as f:
        for b in f:
            line = b.decode("utf-8", "ignore").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get("type")
            qid = obj.get("qid")
            if t is None or qid is None:
                continue
            try:
                qid_i = int(qid)
            except Exception:
                continue
            out[(str(t), qid_i)] = obj
    return out


def _row_from_pred(pred: Dict[str, Any]) -> Dict[str, Any]:
    # Map predictions.jsonl schema -> per_question schema used in this repo
    ans = pred.get("answer")
    abstain = _is_abstain(ans)
    if abstain:
        tp, fp, fn = 1.0, 0.0, 0.0
        judge_method, judge_raw = "abstain", None
    else:
        tp, fp, fn = 0.0, 1.0, 0.0
        judge_method, judge_raw = "not_abstain", None

    precision, recall, f1 = _prf(tp, fp, fn)

    return {
        "type": pred.get("type"),
        "qid": int(pred.get("qid")),
        "question": pred.get("question"),
        "gold": pred.get("gold"),
        "prediction": ans,
        "error": pred.get("error"),
        "confidence": pred.get("confidence"),
        "weak_answer": pred.get("weak_answer"),
        "retrieval_empty": pred.get("retrieval_empty"),
        "num_chunks_retrieved": pred.get("num_chunks_retrieved"),
        "retrieval_time": pred.get("retrieval_time"),
        "generation_time": pred.get("generation_time"),
        "judge_method": judge_method,
        "judge_raw": judge_raw,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "abstain": abstain,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    pred_path = run_dir / "predictions.jsonl"
    per_q_path = run_dir / "per_question.json"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not per_q_path.exists():
        raise FileNotFoundError(per_q_path)

    preds = _read_predictions(pred_path)
    rows: List[Dict[str, Any]] = _load_json(per_q_path)

    existing = {(str(r.get("type")), int(r.get("qid"))) for r in rows}

    to_add: List[Dict[str, Any]] = []
    for qid in range(51, 61):
        key = ("Null", qid)
        if key in existing:
            continue
        pred = preds.get(key)
        if pred is None:
            raise RuntimeError(f"Missing in predictions.jsonl: {key}")
        to_add.append(_row_from_pred(pred))

    if not to_add:
        return 0

    rows.extend(to_add)
    # keep stable ordering by type then qid
    type_order = {"Multi": 0, "Single": 1, "Null": 2}
    rows.sort(key=lambda r: (type_order.get(str(r.get("type")), 99), int(r.get("qid", 0))))

    per_q_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # Recompute all derived artifacts
    from accuracy_test.manual_patch_run import main as patch_main

    patch_main([
        "--run_dir",
        str(run_dir),
        "--error_samples",
        "10",
    ])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

