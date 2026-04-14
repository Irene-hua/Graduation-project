"""Manual patch utility for an existing accuracy_test run folder.

Goal
----
- Patch ONLY specific questions (by type+qid) in an existing run's `per_question.json`
  to force TP/FP/FN (typically set to a correct TP=1).
- Recompute and rewrite all derived artifacts using the same aggregation/report
  logic as `accuracy_test.score_rag_predictions` without re-running RAG or LLM judge.

This is useful for human-in-the-loop verification where the judge LLM mis-scored
obvious correct answers.

Usage (PowerShell)
------------------
python -m accuracy_test.manual_patch_run --run_dir accuracy_test/runs/<run_id> --set-tp Single:1 Single:2 Single:3
python -m accuracy_test.manual_patch_run --run_dir accuracy_test/runs/<run_id> --set-fn Single:28

Notes
-----
- This script updates:
  - per_question.json
  - per_question.csv
  - summary.json
  - report.md
  - error_samples.json / error_samples.md
  - verify_sums.json
  - computed_metrics.json

- It does NOT touch predictions.jsonl.

"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _prf_from_counts(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


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

    out["fp_count"] = sum(1.0 for r in rows if float(r.get("fp", 0.0)) > 0)
    out["fn_count"] = sum(1.0 for r in rows if float(r.get("fn", 0.0)) > 0)

    return out


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def _render_report(out_dir: Path, summary: Dict[str, Any], run_meta: Dict[str, Any]) -> None:
    # Reuse the exact report renderer from score_rag_predictions to stay consistent.
    from accuracy_test.score_rag_predictions import _render_report  # type: ignore

    _render_report(out_dir, summary, run_meta)


def _write_error_samples(out_dir: Path, rows: List[Dict[str, Any]], limit: int = 10) -> None:
    from accuracy_test.score_rag_predictions import _write_error_samples  # type: ignore

    _write_error_samples(out_dir, rows, limit=limit)


def _parse_key(s: str) -> Tuple[str, str]:
    # Accept formats:
    #  - Single:3
    #  - Single/3
    #  - Single,3
    for sep in (":", "/", ","):
        if sep in s:
            t, q = s.split(sep, 1)
            return t.strip(), q.strip()
    raise ValueError(f"Bad key format: {s!r} (expected Type:qid)")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Manual patch an existing scored run folder")
    p.add_argument("--run_dir", type=str, required=True, help="Run directory containing per_question.json")
    p.add_argument(
        "--set-tp",
        nargs="+",
        default=[],
        help="List of Type:qid to force to TP=1,FP=0,FN=0 (e.g., Single:1 Single:2)",
    )
    p.add_argument(
        "--set-fp",
        nargs="+",
        default=[],
        help="List of Type:qid to force to FP=1,TP=0,FN=0 (e.g., Single:30)",
    )
    p.add_argument(
        "--set-fn",
        nargs="+",
        default=[],
        help="List of Type:qid to force to FN=1,TP=0,FP=0 (abstain/拒答; e.g., Single:28)",
    )
    p.add_argument("--error_samples", type=int, default=10, help="How many error samples to export")

    args = p.parse_args(argv)
    run_dir = Path(args.run_dir)

    per_q_path = run_dir / "per_question.json"
    if not per_q_path.exists():
        raise FileNotFoundError(f"Missing {per_q_path}")

    rows: List[Dict[str, Any]] = _load_json(per_q_path)

    tp_targets = set(_parse_key(x) for x in args.set_tp)
    fp_targets = set(_parse_key(x) for x in args.set_fp)
    fn_targets = set(_parse_key(x) for x in args.set_fn)

    overlap = (tp_targets & fp_targets) | (tp_targets & fn_targets) | (fp_targets & fn_targets)
    if overlap:
        raise ValueError(f"Targets can’t overlap between --set-tp/--set-fp/--set-fn: {sorted(overlap)}")

    patched_tp = 0
    patched_fp = 0
    patched_fn = 0

    for r in rows:
        key = (str(r.get("type")), str(r.get("qid")))
        if key in tp_targets:
            r["tp"], r["fp"], r["fn"] = 1.0, 0.0, 0.0
            r["precision"], r["recall"], r["f1"] = 1.0, 1.0, 1.0
            r["abstain"] = False
            # keep an audit trail; doesn't affect metrics
            r["judge_method"] = "manual_override"
            r["judge_raw"] = "forced_tp"
            patched_tp += 1
        elif key in fp_targets:
            r["tp"], r["fp"], r["fn"] = 0.0, 1.0, 0.0
            r["precision"], r["recall"], r["f1"] = 0.0, 0.0, 0.0
            r["abstain"] = False
            r["judge_method"] = "manual_override"
            r["judge_raw"] = "forced_fp"
            patched_fp += 1
        elif key in fn_targets:
            r["tp"], r["fp"], r["fn"] = 0.0, 0.0, 1.0
            r["precision"], r["recall"], r["f1"] = 0.0, 0.0, 0.0
            r["abstain"] = True
            r["judge_method"] = "manual_override"
            r["judge_raw"] = "forced_fn"
            patched_fn += 1

    # ensure all targets were found
    all_keys_in_rows = {(str(r.get("type")), str(r.get("qid"))) for r in rows}
    missing_tp = sorted(tp_targets - all_keys_in_rows)
    missing_fp = sorted(fp_targets - all_keys_in_rows)
    missing_fn = sorted(fn_targets - all_keys_in_rows)
    if missing_tp or missing_fp or missing_fn:
        missing: List[Tuple[str, str]] = []
        missing.extend(missing_tp)
        missing.extend(missing_fp)
        missing.extend(missing_fn)
        raise ValueError(f"Did not find these targets in per_question.json: {missing}")

    # Write per-question artifacts
    _write_json(per_q_path, rows)
    _write_csv(run_dir / "per_question.csv", rows)

    # Recompute summary
    summary: Dict[str, Any] = {
        "Multi": _aggregate([r for r in rows if r.get("type") == "Multi"]),
        "Single": _aggregate([r for r in rows if r.get("type") == "Single"]),
        "Null": _aggregate([r for r in rows if r.get("type") == "Null"]),
        "Overall": _aggregate(rows),
    }

    run_meta_path = run_dir / "run_meta.json"
    run_meta: Dict[str, Any] = {}
    if run_meta_path.exists():
        run_meta = _load_json(run_meta_path) or {}

    summary["run_meta"] = run_meta
    _write_json(run_dir / "summary.json", summary)

    # report + error samples
    _render_report(run_dir, summary, run_meta)
    _write_error_samples(run_dir, rows, limit=int(args.error_samples))

    # verify_sums.json
    verify = {
        "Multi": summary["Multi"]["micro"],
        "Single": summary["Single"]["micro"],
        "Null": summary["Null"]["micro"],
        "Overall": summary["Overall"]["micro"],
    }
    _write_json(run_dir / "verify_sums.json", verify)

    # computed_metrics.json (keep simple + compatible)
    computed = {
        "Multi": summary["Multi"],
        "Single": summary["Single"],
        "Null": summary["Null"],
        "Overall": summary["Overall"],
    }
    _write_json(run_dir / "computed_metrics.json", computed)

    # metrics_check.json (legacy quick-check file used by some scripts)
    metrics_check = {
        "Multi": {
            "n": summary["Multi"]["num_samples"],
            "tp": summary["Multi"]["micro"]["tp"],
            "fp": summary["Multi"]["micro"]["fp"],
            "fn": summary["Multi"]["micro"]["fn"],
        },
        "Single": {
            "n": summary["Single"]["num_samples"],
            "tp": summary["Single"]["micro"]["tp"],
            "fp": summary["Single"]["micro"]["fp"],
            "fn": summary["Single"]["micro"]["fn"],
        },
        "Null": {
            "n": summary["Null"]["num_samples"],
            "tp": summary["Null"]["micro"]["tp"],
            "fp": summary["Null"]["micro"]["fp"],
            "fn": summary["Null"]["micro"]["fn"],
        },
        "Overall": {
            "n": summary["Overall"]["num_samples"],
            "tp": summary["Overall"]["micro"]["tp"],
            "fp": summary["Overall"]["micro"]["fp"],
            "fn": summary["Overall"]["micro"]["fn"],
        },
    }
    _write_json(run_dir / "metrics_check.json", metrics_check)

    changed = []
    if tp_targets:
        changed.append(f"TP:{patched_tp}/{len(tp_targets)}")
    if fp_targets:
        changed.append(f"FP:{patched_fp}/{len(fp_targets)}")
    if fn_targets:
        changed.append(f"FN:{patched_fn}/{len(fn_targets)}")

    print(f"Patched -> {' '.join(changed) if changed else 'none'}")
    om = summary["Overall"]["micro"]
    print(f"Overall micro: P={om['precision']:.4f} R={om['recall']:.4f} F1={om['f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
