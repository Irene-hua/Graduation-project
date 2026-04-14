"""Stage 1: Run RAG in batch and write raw predictions to disk.

This script is intentionally *not* calculating metrics. It only:
- loads Multi/Single/Null datasets
- runs the existing RAG pipeline (no modifications)
- writes one JSON object per question to `predictions.jsonl`

Outputs
-------
accuracy_test/
  runs/<run_id>/
    run_meta.json
    config_patched.yaml (optional; only if local Qdrant storage is isolated)
    predictions.jsonl
    checkpoint.json

Why split stages?
-----------------
Evaluation runs can be long and error-prone (Qdrant local locks, LLM hiccups). By persisting
raw predictions first, we can repeat scoring/report generation offline without re-running RAG.

Usage
-----
python -m accuracy_test.run_rag_predictions --config config/config.yaml --key_file encryption.key --collection_name encrypted_documents_lihua

"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.rag_pipeline.rag_system import _build_runtime


def _load_lines(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_done_ids(pred_path: Path) -> set[Tuple[str, int]]:
    done: set[Tuple[str, int]] = set()
    if not pred_path.exists():
        return done
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                t = str(obj.get("type"))
                qid = int(obj.get("qid"))
                done.add((t, qid))
            except Exception:
                continue
    return done


def _prepare_isolated_qdrant_storage(*, root: Path, out_dir: Path) -> Optional[Path]:
    """Copy local qdrant_storage to an isolated per-run directory.

    Skips `.lock` because it is frequently locked on Windows.
    """

    src = root / "qdrant_storage"
    if not src.exists():
        return None

    dst = out_dir / "qdrant_storage_copy"
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)

    def _ignore_lock(dir_path: str, names: List[str]) -> List[str]:
        return [".lock"] if ".lock" in names else []

    shutil.copytree(src, dst, ignore=_ignore_lock)
    return dst


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage1: batch run RAG and save predictions.jsonl")
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--key_file", type=str, default="encryption.key")
    p.add_argument("--collection_name", type=str, default=None)
    p.add_argument("--allow_empty_collection", action="store_true")

    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_tokens", type=int, default=None)

    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="Resume by skipping already written (type,qid) in predictions.jsonl")

    p.add_argument("--out_dir", type=str, default=None, help="Override output directory")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    root = Path(__file__).resolve().parents[1]

    datasets = [
        ("Multi", root / "data" / "test_datasets" / "lihua-queries1", root / "data" / "gold-answer" / "lihua-queries1-gold-answer"),
        ("Single", root / "data" / "test_datasets" / "lihua-queries2", root / "data" / "gold-answer" / "lihua-queries2-gold-answer"),
        ("Null", root / "data" / "test_datasets" / "lihua-queries3", root / "data" / "gold-answer" / "lihua-queries3-gold-answer"),
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_collection = (args.collection_name or "config").replace("/", "_").replace("\\", "_")
    out_dir = Path(args.out_dir) if args.out_dir else (root / "accuracy_test" / "runs" / f"{timestamp}_pred_{safe_collection}")
    out_dir.mkdir(parents=True, exist_ok=True)

    isolated_storage = _prepare_isolated_qdrant_storage(root=root, out_dir=out_dir)
    effective_config_path = Path(args.config)
    if isolated_storage is not None:
        import yaml

        cfg_obj = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        cfg_obj.setdefault("vector_db", {})
        cfg_obj["vector_db"]["storage_path"] = str(isolated_storage)
        patched_cfg = out_dir / "config_patched.yaml"
        patched_cfg.write_text(yaml.safe_dump(cfg_obj, sort_keys=False, allow_unicode=True), encoding="utf-8")
        effective_config_path = patched_cfg

    rag, _audit_logger = _build_runtime(
        config_path=str(effective_config_path),
        key_file=args.key_file,
        collection_name=args.collection_name,
        allow_empty_collection=args.allow_empty_collection,
    )

    llm_name = getattr(getattr(getattr(rag, "llm", None), "client", None), "model_name", None) or getattr(rag, "llm_name", "unknown")

    pred_path = out_dir / "predictions.jsonl"
    if not args.resume:
        pred_path.write_text("", encoding="utf-8")

    done_ids = _read_done_ids(pred_path) if args.resume else set()

    run_meta = {
        "stage": "predictions",
        "timestamp": timestamp,
        "llm_name": llm_name,
        "collection_name": args.collection_name,
        "config_path": str(effective_config_path),
        "isolated_storage_path": str(isolated_storage) if isolated_storage else None,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "limit": args.limit,
        "resume": bool(args.resume),
        "schema_version": "predictions.v1",
    }
    _write_json(out_dir / "run_meta.json", run_meta)

    planned = 0
    for t, qpath, _gpath in datasets:
        n = len(_load_lines(qpath))
        if args.limit is not None:
            n = min(n, max(0, int(args.limit)))
        planned += n

    written = 0
    for qtype, qpath, gpath in datasets:
        queries = _load_lines(qpath)
        golds = _load_lines(gpath)
        if args.limit is not None:
            queries = queries[: max(0, int(args.limit))]
            golds = golds[: max(0, int(args.limit))]

        for i, (q, gold) in enumerate(zip(queries, golds), start=1):
            if (qtype, i) in done_ids:
                continue

            if written == 0 or written % 5 == 0:
                print(f"[progress] written={written}/{planned} (running {qtype} qid={i})")

            try:
                res = rag.answer_question(
                    question=q,
                    top_k=args.top_k,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except Exception as e:
                res = {"answer": "", "error": str(e)}

            row = {
                "type": qtype,
                "qid": i,
                "question": q,
                "gold": gold,
                "answer": res.get("answer", ""),
                "error": res.get("error"),
                # keep diagnostics if present
                "confidence": res.get("confidence"),
                "weak_answer": res.get("weak_answer"),
                "retrieval_empty": res.get("retrieval_empty"),
                "num_chunks_retrieved": res.get("num_chunks_retrieved"),
                "retrieval_time": res.get("retrieval_time"),
                "generation_time": res.get("generation_time"),
                "used_chunks": res.get("used_chunks"),
            }
            _append_jsonl(pred_path, row)
            written += 1

            if args.sleep:
                time.sleep(float(args.sleep))

            if written % 10 == 0:
                _write_json(out_dir / "checkpoint.json", {"written": written, "planned": planned, "last": {"type": qtype, "qid": i}})

    _write_json(out_dir / "checkpoint.json", {"written": written, "planned": planned, "done": True})
    print(f"Done. predictions.jsonl saved to: {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

