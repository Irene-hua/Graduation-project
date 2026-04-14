#!/usr/bin/env python3
"""Performance comparison on the thesis 60Q datasets (Multi/Single/Null).

This script runs the SAME question sets used by `accuracy_test/run_rag_accuracy_eval.py`, but
focuses on performance overhead of encryption:

- encrypted RAG: existing pipeline + encrypted payloads (ciphertext/nonce)
- plaintext RAG: identical pipeline, but Qdrant payload stores raw text (`text`) in an isolated
  Qdrant storage under `unencrypted/`.

Outputs (paper-ready)
---------------------
unencrypted/results_60q/<run_id>/
  run_meta.json
  samples.jsonl
  samples.csv
  summary.json
  REPORT.md

Notes
-----
- Requires Ollama running.
- Requires BOTH collections populated:
  - encrypted collection: provided via --encrypted_collection (MUST be non-empty)
  - plaintext collection: provided via --plaintext_collection (default created by unencrypted/ingest_plaintext.py)

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def _maybe_import_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None


def _load_lines(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob('*'):
        try:
            if p.is_file():
                total += p.stat().st_size
        except Exception:
            continue
    return int(total)


@dataclass
class Sample:
    system: str               # encrypted|plaintext
    dataset: str              # Multi|Single|Null
    qid: int
    question: str
    retrieval_time_s: float
    generation_time_s: float
    total_time_s: float
    num_chunks: int
    context_length: int
    retrieve_k: int
    ok: bool
    error: Optional[str] = None


def _run_dataset(rag, *, system: str, dataset: str, questions: List[str], sleep_s: float) -> Tuple[List[Sample], Dict]:
    psutil = _maybe_import_psutil()
    proc = psutil.Process(os.getpid()) if psutil else None

    cpu_samples = []
    rss_samples = []

    if proc:
        try:
            proc.cpu_percent(interval=None)
        except Exception:
            pass

    out: List[Sample] = []

    for idx, q in enumerate(questions, start=1):
        if sleep_s:
            time.sleep(float(sleep_s))

        cpu0 = rss0 = None
        if proc:
            try:
                cpu0 = proc.cpu_percent(interval=None)
                rss0 = proc.memory_info().rss
            except Exception:
                cpu0 = rss0 = None

        try:
            res = rag.answer_question(q, top_k=5, temperature=0.2)
        except Exception as e:
            res = {"error": str(e)}

        cpu1 = rss1 = None
        if proc:
            try:
                cpu1 = proc.cpu_percent(interval=None)
                rss1 = proc.memory_info().rss
            except Exception:
                cpu1 = rss1 = None

        if cpu0 is not None and cpu1 is not None:
            cpu_samples.append(max(cpu0, cpu1))
        if rss0 is not None and rss1 is not None:
            rss_samples.append(max(rss0, rss1))

        ok = bool(res.get('path') == 'RAG') if isinstance(res, dict) else False
        out.append(
            Sample(
                system=system,
                dataset=dataset,
                qid=idx,
                question=q,
                retrieval_time_s=float(res.get('retrieval_time') or 0.0) if isinstance(res, dict) else 0.0,
                generation_time_s=float(res.get('generation_time') or 0.0) if isinstance(res, dict) else 0.0,
                total_time_s=float(res.get('total_time') or res.get('time') or 0.0) if isinstance(res, dict) else 0.0,
                num_chunks=int(res.get('num_chunks_retrieved') or 0) if isinstance(res, dict) else 0,
                context_length=int(res.get('context_length') or 0) if isinstance(res, dict) else 0,
                retrieve_k=int(res.get('retrieve_k') or 0) if isinstance(res, dict) else 0,
                ok=ok,
                error=res.get('error') if isinstance(res, dict) else None,
            )
        )
        if idx == 1 or idx % 10 == 0:
            print(f"[perf] system={system} dataset={dataset} progress={idx}/{len(questions)}")

    res_stats = {
        'psutil_available': bool(psutil),
        'cpu_percent_avg': (sum(cpu_samples) / len(cpu_samples)) if cpu_samples else None,
        'cpu_percent_max': max(cpu_samples) if cpu_samples else None,
        'rss_bytes_avg': (sum(rss_samples) / len(rss_samples)) if rss_samples else None,
        'rss_bytes_max': max(rss_samples) if rss_samples else None,
    }
    return out, res_stats


def _append_jsonl(path: Path, rows: List[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        for s in rows:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + '\n')


def _summarize(df: pd.DataFrame) -> Dict:
    def q(x, p):
        return float(x.quantile(p)) if len(x) else None

    return {
        'n': int(len(df)),
        'ok_rate': float((df['ok'] == True).mean()) if len(df) else 0.0,
        'retrieval_time_s': {
            'mean': float(df['retrieval_time_s'].mean()),
            'median': float(df['retrieval_time_s'].median()),
            'p90': q(df['retrieval_time_s'], 0.90),
            'p95': q(df['retrieval_time_s'], 0.95),
        },
        'generation_time_s': {
            'mean': float(df['generation_time_s'].mean()),
            'median': float(df['generation_time_s'].median()),
            'p90': q(df['generation_time_s'], 0.90),
            'p95': q(df['generation_time_s'], 0.95),
        },
        'total_time_s': {
            'mean': float(df['total_time_s'].mean()),
            'median': float(df['total_time_s'].median()),
            'p90': q(df['total_time_s'], 0.90),
            'p95': q(df['total_time_s'], 0.95),
        },
    }


def _fmt_bytes(v: Optional[float]) -> str:
    if v is None:
        return 'n/a'
    v = float(v)
    gb = v / (1024 ** 3)
    mb = v / (1024 ** 2)
    return f"{gb:.3f} GiB" if gb >= 1 else f"{mb:.1f} MiB"


def _md_report(summary: Dict) -> str:
    lines: List[str] = []
    lines.append('# 加密RAG vs 明文RAG：60题三测试集性能对比报告')
    lines.append('')
    lines.append(f"- run_id: `{summary['run_id']}`")
    lines.append(f"- timestamp: `{summary['timestamp']}`")
    lines.append(f"- encrypted_collection: `{summary['encrypted']['collection_name']}`")
    lines.append(f"- plaintext_collection: `{summary['plaintext']['collection_name']}`")
    lines.append('')

    lines.append('## 资源/存储概览')
    lines.append('| system | storage_path | qdrant_dir_size | points | rss_max | cpu_max |')
    lines.append('|---|---|---:|---:|---:|---:|')
    for sys_name in ['encrypted', 'plaintext']:
        s = summary[sys_name]
        lines.append(
            f"| {sys_name} | `{s['storage_path']}` | {_fmt_bytes(s['storage_size_bytes'])} | {s.get('points_count','n/a')} | {_fmt_bytes(s['resources'].get('rss_bytes_max'))} | {s['resources'].get('cpu_percent_max','n/a')} |"
        )
    lines.append('')

    lines.append('## 分测试集延迟统计（median/p95）')
    lines.append('| dataset | metric | encrypted median | encrypted p95 | plaintext median | plaintext p95 | ratio(median) |')
    lines.append('|---|---|---:|---:|---:|---:|---:|')

    for dataset in ['Multi', 'Single', 'Null']:
        for metric in ['retrieval_time_s', 'generation_time_s', 'total_time_s']:
            e = summary['encrypted']['datasets'][dataset]['latency'][metric]
            p = summary['plaintext']['datasets'][dataset]['latency'][metric]
            em = e.get('median')
            pm = p.get('median')
            ratio = (em / pm) if (em is not None and pm not in (None, 0)) else None
            lines.append(
                f"| {dataset} | {metric} | {em:.4f} | {e.get('p95'):.4f} | {pm:.4f} | {p.get('p95'):.4f} | {('n/a' if ratio is None else f'{ratio:.3f}x')} |"
            )

    lines.append('')
    lines.append('## 结论写作要点（可直接用于论文的小结结构）')
    lines.append('1. **检索阶段**：对比 `retrieval_time_s`，可直接量化“加密payload+解密流程”在 Multi/Single/Null 三类问题下的额外开销。')
    lines.append('2. **生成阶段**：对比 `generation_time_s`，通常两者接近，说明加密主要影响检索与上下文构建阶段。')
    lines.append('3. **资源与存储**：对比 Qdrant 目录大小与 RSS 峰值，讨论 payload 膨胀与解密过程对内存的影响。')
    lines.append('')
    lines.append('---')
    lines.append('本报告由 `unencrypted/bench/run_perf_compare_60q.py` 自动生成。')
    lines.append('')
    return '\n'.join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Run 60Q perf compare: encrypted vs plaintext, Multi/Single/Null')
    ap.add_argument('--config', default='config/config.yaml')
    ap.add_argument('--key_file', default='encryption.key')

    ap.add_argument('--encrypted_collection', required=True, help='Encrypted non-empty collection (e.g., encrypted_documents_lihua)')
    ap.add_argument('--encrypted_storage', default=None, help='Encrypted storage path override (default from config.yaml)')

    ap.add_argument('--plaintext_collection', default='plaintext_documents_lihua_world')
    ap.add_argument('--plaintext_storage', default='unencrypted/qdrant_storage_plaintext')

    ap.add_argument('--sleep_s', type=float, default=0.0)
    ap.add_argument('--limit', type=int, default=None, help='Optional limit per dataset (debug)')

    args = ap.parse_args(argv)

    import yaml

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()

    config = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))

    encrypted_storage = args.encrypted_storage or config['vector_db']['storage_path']

    from src.rag_pipeline.rag_system import _build_runtime
    from unencrypted.build_plaintext_rag import build_plaintext_runtime

    # Build encrypted runtime from a patched in-memory config by writing a temporary cfg
    patched_cfg = dict(config)
    patched_cfg.setdefault('vector_db', {})
    patched_cfg['vector_db']['storage_path'] = encrypted_storage

    tmp_cfg_path = (PROJECT_ROOT / 'unencrypted' / 'bench' / '_tmp_cfg_encrypted.yaml').resolve()
    tmp_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg_path.write_text(yaml.safe_dump(patched_cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')

    encrypted_rag, _ = _build_runtime(
        config_path=str(tmp_cfg_path),
        key_file=args.key_file,
        collection_name=args.encrypted_collection,
        allow_empty_collection=False,
    )

    plaintext_rag, _ = build_plaintext_runtime(
        config_path=str(cfg_path),
        key_file=args.key_file,
        collection_name=args.plaintext_collection,
        storage_path=args.plaintext_storage,
        allow_empty_collection=False,
    )

    datasets = {
        'Multi': (PROJECT_ROOT / 'data' / 'test_datasets' / 'lihua-queries1'),
        'Single': (PROJECT_ROOT / 'data' / 'test_datasets' / 'lihua-queries2'),
        'Null': (PROJECT_ROOT / 'data' / 'test_datasets' / 'lihua-queries3'),
    }

    questions_by_type: Dict[str, List[str]] = {}
    for k, p in datasets.items():
        qs = _load_lines(p)
        if args.limit is not None:
            qs = qs[: max(0, int(args.limit))]
        questions_by_type[k] = qs

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = (PROJECT_ROOT / 'unencrypted' / 'results_60q' / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write run_meta early so partial runs are still traceable.
    run_meta = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'encrypted_collection': args.encrypted_collection,
        'plaintext_collection': args.plaintext_collection,
        'encrypted_storage': encrypted_storage,
        'plaintext_storage': args.plaintext_storage,
        'sleep_s': args.sleep_s,
        'limit': args.limit,
    }
    (out_dir / 'run_meta.json').write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding='utf-8')

    # Storage footprints
    enc_storage_path = (PROJECT_ROOT / encrypted_storage).resolve() if not Path(encrypted_storage).is_absolute() else Path(encrypted_storage)
    pl_storage_path = (PROJECT_ROOT / args.plaintext_storage).resolve() if not Path(args.plaintext_storage).is_absolute() else Path(args.plaintext_storage)

    summary: Dict = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'config': {
            'embedding_model': config['embedding']['model_name'],
            'distance_metric': config['vector_db']['distance_metric'],
            'rerank_enabled': bool(config.get('rerank', {}).get('enabled', False)),
            'llm_model': config.get('llm', {}).get('default_model') or config['llm'].get('model_name', 'mistral'),
            'llm_base_url': config['llm']['base_url'],
            'sleep_s': args.sleep_s,
            'limit': args.limit,
        },
        'encrypted': {
            'collection_name': args.encrypted_collection,
            'storage_path': str(enc_storage_path),
            'storage_size_bytes': _dir_size_bytes(enc_storage_path),
            'points_count': encrypted_rag.retriever.vector_store.get_collection_info().get('points_count'),
            'datasets': {},
            'resources': {},
        },
        'plaintext': {
            'collection_name': args.plaintext_collection,
            'storage_path': str(pl_storage_path),
            'storage_size_bytes': _dir_size_bytes(pl_storage_path),
            'points_count': plaintext_rag.retriever.vector_store.get_collection_info().get('points_count'),
            'datasets': {},
            'resources': {},
        },
    }

    all_samples: List[Sample] = []
    samples_jsonl = out_dir / 'samples.jsonl'
    samples_csv = out_dir / 'samples.csv'
    # reset artifacts
    if samples_jsonl.exists():
        samples_jsonl.unlink()

    # Run encrypted then plaintext for each dataset to keep conditions similar.
    try:
        for dataset, qs in questions_by_type.items():
            enc_samples, enc_res = _run_dataset(encrypted_rag, system='encrypted', dataset=dataset, questions=qs, sleep_s=args.sleep_s)
            _append_jsonl(samples_jsonl, enc_samples)

            pl_samples, pl_res = _run_dataset(plaintext_rag, system='plaintext', dataset=dataset, questions=qs, sleep_s=args.sleep_s)
            _append_jsonl(samples_jsonl, pl_samples)

            all_samples.extend(enc_samples)
            all_samples.extend(pl_samples)

            enc_df = pd.DataFrame([asdict(s) for s in enc_samples])
            pl_df = pd.DataFrame([asdict(s) for s in pl_samples])

            summary['encrypted']['datasets'][dataset] = {
                'latency': _summarize(enc_df),
            }
            summary['plaintext']['datasets'][dataset] = {
                'latency': _summarize(pl_df),
            }

            # merge resource stats (max across datasets)
            def _merge_resources(dst: Dict, src: Dict):
                for k in ['cpu_percent_avg', 'cpu_percent_max', 'rss_bytes_avg', 'rss_bytes_max']:
                    v = src.get(k)
                    if v is None:
                        continue
                    if dst.get(k) is None:
                        dst[k] = v
                    else:
                        dst[k] = max(dst[k], v) if 'max' in k else (dst[k] + v) / 2
                dst['psutil_available'] = dst.get('psutil_available') or src.get('psutil_available')

            _merge_resources(summary['encrypted']['resources'], enc_res)
            _merge_resources(summary['plaintext']['resources'], pl_res)

            # Write a rolling CSV snapshot so interruption still yields data.
            pd.DataFrame([asdict(s) for s in all_samples]).to_csv(samples_csv, index=False, encoding='utf-8')
    finally:
        # Best-effort finalize report even on interruption.
        if all_samples:
            pd.DataFrame([asdict(s) for s in all_samples]).to_csv(samples_csv, index=False, encoding='utf-8')
        (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        (out_dir / 'REPORT.md').write_text(_md_report(summary), encoding='utf-8')

    print(f"Wrote: {out_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
