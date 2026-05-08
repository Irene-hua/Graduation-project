"""Rescore Experiment 1 (compare/*_scored.json) with Experiment 2's scoring rules.

Goal
----
- Take three Experiment-1 JSON outputs (Single/Multi/Null) under `compare/`.
- Extract ONLY the Mistral answers.
- Convert them into `predictions.jsonl` compatible with `accuracy_test/score_rag_predictions.py`.
- Run the Stage-2 scoring script to compute P/R/F1 with identical abstain & judge rules.
- Write ALL artifacts under `BAcompare/`.

This script is intentionally self-contained and uses only the local repo code.

Run (PowerShell)
---------------
python BAcompare/rescore_compare_with_accuracy_test.py

Notes
-----
- The scorer uses local Ollama (`ollama run <judge_model>`) for semantic matching. If Ollama
  isn't available, it falls back to a deterministic substring matcher (still consistent with the scorer).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_DIR = REPO_ROOT / "compare"
ACC_RUN_SUMMARY = (
    REPO_ROOT
    / "accuracy_test"
    / "runs"
    / "test_encrypted_documents_lihua"
    / "summary.json"
)


@dataclass(frozen=True)
class InputSpec:
    name: str  # Single/Multi/Null
    scored_path: Path


SPECS: List[InputSpec] = [
    InputSpec("Single", COMPARE_DIR / "llm_compare_20260401_131033_scored.json"),
    InputSpec("Multi", COMPARE_DIR / "llm_compare_20260401_154508_Multi_scored.json"),
    InputSpec("Null", COMPARE_DIR / "llm_compare_20260401_184038_Null_scored.json"),
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_mistral_rows(scored: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    items = scored.get("items") or []
    for it in items:
        # Minimal contract consistent with scorer expectations.
        yield {
            "type": it.get("type") or (scored.get("meta") or {}).get("type"),
            "qid": it.get("idx"),
            "question": it.get("question"),
            "gold": it.get("ground_truth"),
            # IMPORTANT: Stage-2 scorer reads `answer`, not `prediction`.
            "answer": ((it.get("mistral") or {}).get("answer") or ""),
        }


def _write_predictions_jsonl(out_path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            # Ensure JSON-serializable and stable.
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _run_scorer(predictions_path: Path, out_dir: Path, judge_model: str = "llama3.2:3b") -> None:
    """Run accuracy_test.score_rag_predictions in a subprocess.

    The Stage-2 scorer writes outputs next to `predictions.jsonl` by default.
    To keep everything under `BAcompare/`, we run it in-place inside `out_dir`.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Score script writes next to the predictions file, so keep predictions in out_dir.
    if predictions_path.parent.resolve() != out_dir.resolve():
        raise ValueError("predictions_path must live inside out_dir")

    cmd = [
        "python",
        "-m",
        "accuracy_test.score_rag_predictions",
        "--predictions",
        str(predictions_path),
        "--judge_model",
        judge_model,
    ]

    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _copy_input_for_audit(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    out_root = REPO_ROOT / "BAcompare" / "exp1_rescored_with_exp2_rules_20260508"
    out_root.mkdir(parents=True, exist_ok=True)

    # Capture experiment-2 baseline (already computed in repo) for comparison.
    exp2_summary: Optional[Dict[str, Any]] = None
    if ACC_RUN_SUMMARY.exists():
        exp2_summary = _load_json(ACC_RUN_SUMMARY)
        _dump_json(out_root / "exp2_existing_summary.json", exp2_summary)

    results: Dict[str, Any] = {
        "exp2_existing_summary_path": str(ACC_RUN_SUMMARY) if ACC_RUN_SUMMARY.exists() else None,
        "exp1_rescored": {},
    }

    # Keep paths for merge step
    per_type_pred_paths: Dict[str, Path] = {}

    for spec in SPECS:
        scored = _load_json(spec.scored_path)
        run_tag = (scored.get("meta") or {}).get("run_tag") or spec.scored_path.stem
        subdir = out_root / f"{run_tag}_{spec.name}"
        subdir.mkdir(parents=True, exist_ok=True)

        _copy_input_for_audit(spec.scored_path, subdir / "input_scored.json")

        # Convert to predictions.jsonl compatible with accuracy_test scorer.
        pred_path = subdir / "predictions.jsonl"
        n = _write_predictions_jsonl(pred_path, _iter_mistral_rows(scored))
        per_type_pred_paths[spec.name] = pred_path

        # Run scorer (writes per_question.*, summary.json, report.md, etc.).
        _run_scorer(pred_path, subdir)

        summary_path = subdir / "summary.json"
        summary = _load_json(summary_path) if summary_path.exists() else None

        results["exp1_rescored"][spec.name] = {
            "source_scored_path": str(spec.scored_path),
            "out_dir": str(subdir),
            "num_rows": n,
            "summary": summary,
        }

    # Merge 3x60 => 180, rescore once to get the unified "优化前 Overall".
    merged_dir = out_root / "llm_compare_20260401_merged_Overall180"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_pred_path = merged_dir / "predictions.jsonl"

    merged_rows: List[Dict[str, Any]] = []
    for t in ("Multi", "Single", "Null"):
        p = per_type_pred_paths.get(t)
        if not p or not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                merged_rows.append(json.loads(line))

    _write_predictions_jsonl(merged_pred_path, merged_rows)
    _run_scorer(merged_pred_path, merged_dir)

    merged_summary_path = merged_dir / "summary.json"
    merged_summary = _load_json(merged_summary_path) if merged_summary_path.exists() else None
    results["exp1_rescored"]["Overall180"] = {
        "out_dir": str(merged_dir),
        "num_rows": len(merged_rows),
        "summary": merged_summary,
    }

    _dump_json(out_root / "exp1_rescore_results.json", results)

    # Create a small markdown comparison table.
    lines: List[str] = []
    lines.append("# 实验一（compare）用实验二评分口径重评估：Mistral F1 对比\n")
    lines.append("\n## 实验二（accuracy_test）已有结果（mistral）\n")
    if exp2_summary:
        overall = exp2_summary.get("Overall", {}).get("micro", {})
        lines.append(
            f"- Overall micro F1: **{overall.get('f1'):.6f}** (P={overall.get('precision'):.6f}, R={overall.get('recall'):.6f})\n"
        )
    else:
        lines.append("- 未找到实验二 summary.json（请检查 accuracy_test/runs/...）\n")

    lines.append("\n## 实验一重评估结果（使用 accuracy_test/score_rag_predictions.py 口径，仅 mistral）\n")
    lines.append("| 子集 | micro P | micro R | micro F1 | TP | FP | FN | 输出目录 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|\n")

    for name in ["Multi", "Single", "Null"]:
        # Each rescored summary contains Multi/Single/Null/Overall keys; we pick the matching key.
        s = ((results["exp1_rescored"].get(name) or {}).get("summary") or {}).get(name, {})
        micro = s.get("micro", {})
        if micro:
            lines.append(
                "| {name} | {p:.6f} | {r:.6f} | {f1:.6f} | {tp:.0f} | {fp:.0f} | {fn:.0f} | `{out}` |\n".format(
                    name=name,
                    p=float(micro.get("precision", 0.0)),
                    r=float(micro.get("recall", 0.0)),
                    f1=float(micro.get("f1", 0.0)),
                    tp=float(micro.get("tp", 0.0)),
                    fp=float(micro.get("fp", 0.0)),
                    fn=float(micro.get("fn", 0.0)),
                    out=str((results["exp1_rescored"].get(name) or {}).get("out_dir")),
                )
            )
        else:
            lines.append(f"| {name} | (no data) | (no data) | (no data) |  |  |  |  |\n")

    # Append unified 180-question overall row (from merged run's Overall)
    merged_overall_micro = (((results["exp1_rescored"].get("Overall180") or {}).get("summary") or {}).get("Overall") or {}).get("micro", {})
    merged_out_dir = (results["exp1_rescored"].get("Overall180") or {}).get("out_dir")
    if merged_overall_micro:
        lines.append(
            "| Overall(180) | {p:.6f} | {r:.6f} | {f1:.6f} | {tp:.0f} | {fp:.0f} | {fn:.0f} | `{out}` |\n".format(
                p=float(merged_overall_micro.get("precision", 0.0)),
                r=float(merged_overall_micro.get("recall", 0.0)),
                f1=float(merged_overall_micro.get("f1", 0.0)),
                tp=float(merged_overall_micro.get("tp", 0.0)),
                fp=float(merged_overall_micro.get("fp", 0.0)),
                fn=float(merged_overall_micro.get("fn", 0.0)),
                out=str(merged_out_dir),
            )
        )

    (out_root / "EXP1_vs_EXP2_MISTRAL_F1.md").write_text("".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
