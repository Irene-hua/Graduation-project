"""Patch per_question.json for specific qids and regenerate summary/report artifacts.

Designed for Windows PowerShell usage.

Usage:
  python scripts/patch_eval_run.py --run_dir accuracy_test/runs/20260411_201333_pred_encrypted_documents_lihua

It will:
  - set Multi qid=48 to forced_fn (tp=0, fn=1, abstain=True)
  - set Null qid=28 to forced_tp (tp=1, fn=0, abstain=True)
  - (optionally) keep other fields intact
  - write per_question.json
  - call accuracy_test.manual_patch_run to regenerate summary.json/report.md/verify_sums.json
  - print aggregates
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    per_path = run_dir / "per_question.json"
    if not per_path.exists():
        print(f"per_question.json not found: {per_path}", file=sys.stderr)
        return 2

    rows = json.loads(per_path.read_text(encoding="utf-8"))

    def patch_multi48(r: dict) -> None:
        r["judge_method"] = "manual_override"
        r["judge_raw"] = "forced_fn"
        r["tp"] = 0.0
        r["fp"] = 0.0
        r["fn"] = 1.0
        r["precision"] = 0.0
        r["recall"] = 0.0
        r["f1"] = 0.0
        r["abstain"] = True

    def patch_null28(r: dict) -> None:
        r["judge_method"] = "manual_override"
        r["judge_raw"] = "forced_tp"
        r["tp"] = 1.0
        r["fp"] = 0.0
        r["fn"] = 0.0
        r["precision"] = 1.0
        r["recall"] = 1.0
        r["f1"] = 1.0
        r["abstain"] = True

    found = {"Multi48": 0, "Null28": 0}
    for r in rows:
        if r.get("type") == "Multi" and r.get("qid") == 48:
            patch_multi48(r)
            found["Multi48"] += 1
        if r.get("type") == "Null" and r.get("qid") == 28:
            patch_null28(r)
            found["Null28"] += 1

    if found["Multi48"] != 1 or found["Null28"] != 1:
        print(f"Unexpected counts: {found}", file=sys.stderr)
        return 3

    per_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # regenerate artifacts
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "accuracy_test.manual_patch_run",
            "--run_dir",
            str(run_dir),
            "--error_samples",
            "10",
        ]
    )

    # compute aggregates from per_question.json
    rows2 = json.loads(per_path.read_text(encoding="utf-8"))
    agg = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "n": 0})
    for r in rows2:
        t = r["type"]
        agg[t]["tp"] += float(r.get("tp", 0.0))
        agg[t]["fp"] += float(r.get("fp", 0.0))
        agg[t]["fn"] += float(r.get("fn", 0.0))
        agg[t]["n"] += 1

    for name in ["Multi", "Single", "Null"]:
        d = agg[name]
        p, r, f = prf(d["tp"], d["fp"], d["fn"])
        print(f"{name}: P={p:.4f}, R={r:.4f}, F1={f:.4f} (TP={d['tp']:.0f}, FP={d['fp']:.0f}, FN={d['fn']:.0f}, n={d['n']})")

    tp = agg["Multi"]["tp"] + agg["Single"]["tp"] + agg["Null"]["tp"]
    fp = agg["Multi"]["fp"] + agg["Single"]["fp"] + agg["Null"]["fp"]
    fn = agg["Multi"]["fn"] + agg["Single"]["fn"] + agg["Null"]["fn"]
    n = agg["Multi"]["n"] + agg["Single"]["n"] + agg["Null"]["n"]
    p, r, f = prf(tp, fp, fn)
    print(f"Overall: P={p:.4f}, R={r:.4f}, F1={f:.4f} (TP={tp:.0f}, FP={fp:.0f}, FN={fn:.0f}, n={n})")

    # print the patched rows
    m48 = next(x for x in rows2 if x.get("type") == "Multi" and x.get("qid") == 48)
    n28 = next(x for x in rows2 if x.get("type") == "Null" and x.get("qid") == 28)
    print("Multi48_row", {k: m48.get(k) for k in ["judge_raw", "tp", "fp", "fn", "abstain"]})
    print("Null28_row", {k: n28.get(k) for k in ["judge_raw", "tp", "fp", "fn", "abstain"]})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

