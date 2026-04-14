#!/usr/bin/env python3
"""Performance comparison: encrypted RAG vs plaintext baseline.

This script is designed to be Paper-ready:
- records experiment config
- collects latency distributions (retrieval, generation, total)
- samples process CPU+RSS during queries
- measures Qdrant storage directory size as a proxy for index/db footprint
- outputs JSON/CSV/Markdown report under `unencrypted/results/<run_id>/`

Assumptions
- Encrypted pipeline has already ingested to the default collection configured in `config/config.yaml`
  (or you can override with --encrypted_collection).
- Plaintext baseline has already ingested via `unencrypted/ingest_plaintext.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure running this file directly works (so `import src...` resolves).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


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
class PhaseSample:
    system: str  # 'encrypted'|'plaintext'
    query_id: int
    query: str
    retrieval_time_s: float
    generation_time_s: float
    total_time_s: float
    num_chunks: int
    context_length: int
    retrieve_k: int
    ok: bool
    error: Optional[str] = None


def _maybe_import_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None


def _run_queries(rag, queries: List[str], system_name: str, *, sleep_s: float = 0.0) -> Tuple[List[PhaseSample], Dict]:
    psutil = _maybe_import_psutil()
    proc = psutil.Process(os.getpid()) if psutil else None

    samples: List[PhaseSample] = []

    cpu_samples = []
    rss_samples = []

    # Prime CPU percent measurement
    if proc:
        try:
            proc.cpu_percent(interval=None)
        except Exception:
            pass

    for i, q in enumerate(queries):
        if sleep_s:
            time.sleep(float(sleep_s))

        cpu0 = None
        rss0 = None
        if proc:
            try:
                cpu0 = proc.cpu_percent(interval=None)
                rss0 = proc.memory_info().rss
            except Exception:
                cpu0, rss0 = None, None

        ok = True
        err = None
        try:
            out = rag.answer_question(q, top_k=5, temperature=0.2)
        except Exception as e:
            out = None
            ok = False
            err = str(e)

        cpu1 = None
        rss1 = None
        if proc:
            try:
                cpu1 = proc.cpu_percent(interval=None)
                rss1 = proc.memory_info().rss
            except Exception:
                cpu1, rss1 = None, None

        if cpu0 is not None and cpu1 is not None:
            cpu_samples.append(max(cpu0, cpu1))
        if rss0 is not None and rss1 is not None:
            rss_samples.append(max(rss0, rss1))

        if out is None:
            samples.append(
                PhaseSample(
                    system=system_name,
                    query_id=i,
                    query=q,
                    retrieval_time_s=0.0,
                    generation_time_s=0.0,
                    total_time_s=0.0,
                    num_chunks=0,
                    context_length=0,
                    retrieve_k=0,
                    ok=False,
                    error=err,
                )
            )
            continue

        samples.append(
            PhaseSample(
                system=system_name,
                query_id=i,
                query=q,
                retrieval_time_s=float(out.get('retrieval_time') or 0.0),
                generation_time_s=float(out.get('generation_time') or 0.0),
                total_time_s=float(out.get('total_time') or out.get('time') or 0.0),
                num_chunks=int(out.get('num_chunks_retrieved') or 0),
                context_length=int(out.get('context_length') or 0),
                retrieve_k=int(out.get('retrieve_k') or 0),
                ok=bool(out.get('path') == 'RAG'),
                error=out.get('error'),
            )
        )

    stats = {
        'cpu_percent_max': max(cpu_samples) if cpu_samples else None,
        'cpu_percent_avg': (sum(cpu_samples) / len(cpu_samples)) if cpu_samples else None,
        'rss_bytes_max': max(rss_samples) if rss_samples else None,
        'rss_bytes_avg': (sum(rss_samples) / len(rss_samples)) if rss_samples else None,
        'psutil_available': bool(psutil),
    }
    return samples, stats


def _summarize(df) -> Dict:
    def q(x, p):
        return float(x.quantile(p)) if len(x) else None

    out = {}
    for col in ['retrieval_time_s', 'generation_time_s', 'total_time_s']:
        s = df[col]
        out[col] = {
            'mean': float(s.mean()),
            'median': float(s.median()),
            'p90': q(s, 0.90),
            'p95': q(s, 0.95),
            'min': float(s.min()),
            'max': float(s.max()),
        }
    out['ok_rate'] = float((df['ok'] == True).mean())
    return out


def _md_report(summary: Dict) -> str:
    enc = summary['encrypted']
    pl = summary['plaintext']

    def fmt_s(v):
        return 'n/a' if v is None else f"{v:.4f}"

    def fmt_bytes(v):
        if v is None:
            return 'n/a'
        gb = v / (1024 ** 3)
        mb = v / (1024 ** 2)
        if gb >= 1:
            return f"{gb:.3f} GiB"
        return f"{mb:.1f} MiB"

    lines = []
    lines.append('# 加密RAG vs 明文RAG 性能对比实验报告\n')
    lines.append(f"- run_id: `{summary['run_id']}`")
    lines.append(f"- timestamp: `{summary['timestamp']}`")
    lines.append(f"- dataset: `{summary['dataset']}`")
    lines.append('')

    lines.append('## 实验设置')
    for k, v in summary['config'].items():
        lines.append(f"- {k}: {v}")
    lines.append('')

    lines.append('## 数据库/索引占用（Qdrant 本地存储目录大小近似）')
    lines.append('| system | storage_path | collection | size | points |')
    lines.append('|---|---|---:|---:|---:|')
    for sys_name in ['encrypted', 'plaintext']:
        d = summary['storage'][sys_name]
        lines.append(f"| {sys_name} | `{d['storage_path']}` | `{d['collection_name']}` | {fmt_bytes(d['storage_size_bytes'])} | {d.get('points_count','n/a')} |")
    lines.append('')

    lines.append('## 端到端延迟（answer_question）')
    lines.append('| metric | encrypted | plaintext | delta (enc - plain) | slow_down(enc/plain) |')
    lines.append('|---|---:|---:|---:|---:|')
    for col in ['retrieval_time_s', 'generation_time_s', 'total_time_s']:
        em = enc['latency'][col]['median']
        pm = pl['latency'][col]['median']
        delta = (em - pm) if (em is not None and pm is not None) else None
        ratio = (em / pm) if (em is not None and pm not in (None, 0)) else None
        lines.append(
            f"| {col} median (s) | {fmt_s(em)} | {fmt_s(pm)} | {fmt_s(delta)} | {'n/a' if ratio is None else f'{ratio:.3f}x'} |"
        )
    lines.append('')

    lines.append('## 资源占用（脚本进程采样，需 psutil）')
    lines.append('| system | cpu_percent_avg | cpu_percent_max | rss_avg | rss_max |')
    lines.append('|---|---:|---:|---:|---:|')
    for sys_name, d in [('encrypted', enc['resources']), ('plaintext', pl['resources'])]:
        lines.append(
            f"| {sys_name} | {d.get('cpu_percent_avg','n/a')} | {d.get('cpu_percent_max','n/a')} | {fmt_bytes(d.get('rss_bytes_avg'))} | {fmt_bytes(d.get('rss_bytes_max'))} |"
        )
    lines.append('')

    lines.append('## 结论（自动生成，需结合你的运行结果确认）')
    lines.append('- 如果 `retrieval_time_s` 的中位数在 encrypted 明显更高，可归因于：payload 解密开销 + Python层解密循环 + 额外的字段处理。')
    lines.append('- 如果 `generation_time_s` 两者接近，说明加密机制主要影响检索阶段，而对 LLM 生成阶段影响不大。')
    lines.append('- 如果 Qdrant 存储目录大小在 encrypted 更大，可能因为密文长度更长、payload 增大导致存储膨胀。')
    lines.append('')

    lines.append('---')
    lines.append('附：本报告由 `unencrypted/bench/run_perf_compare.py` 自动生成。')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/config.yaml')
    ap.add_argument('--key_file', default='encryption.key')

    ap.add_argument('--dataset', default='data/raw/LiHua-World')

    ap.add_argument('--plaintext_collection', default='plaintext_documents_lihua_world')
    ap.add_argument('--plaintext_storage', default='unencrypted/qdrant_storage_plaintext')

    ap.add_argument('--encrypted_collection', default=None, help='Override encrypted collection name (else use config)')
    ap.add_argument('--encrypted_storage', default=None, help='Override encrypted storage path (else use config)')

    ap.add_argument('--queries_file', default='unencrypted/bench/queries_lihua_world.json')
    ap.add_argument('--n', type=int, default=30, help='Number of queries to run (from queries_file)')
    ap.add_argument('--sleep_s', type=float, default=0.0, help='Sleep between queries to reduce thermal noise')

    args = ap.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent.parent

    # Load config for encrypted defaults
    cfg = Path(args.config)
    if not cfg.is_absolute():
        cfg = (project_root / cfg).resolve()
    with open(cfg, 'r', encoding='utf-8') as f:
        config = json.loads(json.dumps(__import__('yaml').safe_load(f)))

    enc_collection = args.encrypted_collection or config['vector_db']['collection_name']
    enc_storage = args.encrypted_storage or config['vector_db']['storage_path']

    def _pick_non_empty_encrypted_collection(collection_name: str) -> str:
        """If the configured encrypted collection is empty, pick a non-empty encrypted_* collection.

        This keeps the encrypted system untouched while allowing the benchmark to run in workspaces
        where multiple encrypted collections exist.
        """
        try:
            from src.embedding import EmbeddingModel
            from src.retrieval import VectorStore

            embedding_model = EmbeddingModel(model_name=config['embedding']['model_name'])
            # Local-only: use the same storage path configured for the encrypted system.
            vs = VectorStore(
                collection_name=collection_name,
                dimension=embedding_model.get_dimension(),
                distance_metric=config['vector_db']['distance_metric'],
                storage_path=enc_storage,
                host=config['vector_db'].get('host'),
                port=config['vector_db'].get('port'),
            )
            info = vs.get_collection_info()
            if int(info.get('points_count') or 0) > 0:
                return collection_name

            # Scan collections and pick first non-empty encrypted_* one.
            cols = [c.name for c in vs.client.get_collections().collections]
            for name in cols:
                if not str(name).startswith('encrypted'):
                    continue
                try:
                    vs2 = VectorStore(
                        collection_name=name,
                        dimension=embedding_model.get_dimension(),
                        distance_metric=config['vector_db']['distance_metric'],
                        storage_path=enc_storage,
                        host=config['vector_db'].get('host'),
                        port=config['vector_db'].get('port'),
                    )
                    if int(vs2.get_collection_info().get('points_count') or 0) > 0:
                        return name
                except Exception:
                    continue
        except Exception:
            return collection_name
        return collection_name

    enc_collection = _pick_non_empty_encrypted_collection(enc_collection)

    # Ensure _build_runtime points at the same storage path used for collection scanning.
    # We do NOT modify any files; this is an in-memory override for this benchmark run only.
    try:
        config['vector_db']['storage_path'] = enc_storage
    except Exception:
        pass

    # Build both runtimes
    from src.rag_pipeline.rag_system import _build_runtime
    from unencrypted.build_plaintext_rag import build_plaintext_runtime

    encrypted_rag, _ = _build_runtime(
        config_path=str(cfg),
        key_file=str((project_root / args.key_file).resolve() if not Path(args.key_file).is_absolute() else args.key_file),
        collection_name=enc_collection,
        allow_empty_collection=False,
    )

    plaintext_rag, _ = build_plaintext_runtime(
        config_path=str(cfg),
        key_file=str((project_root / args.key_file).resolve() if not Path(args.key_file).is_absolute() else args.key_file),
        collection_name=args.plaintext_collection,
        storage_path=args.plaintext_storage,
        allow_empty_collection=False,
    )

    # Queries
    qf = Path(args.queries_file)
    if not qf.is_absolute():
        qf = (project_root / qf).resolve()
    with open(qf, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    queries = list(queries)[: int(args.n)]

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = (project_root / 'unencrypted' / 'results' / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Storage sizes
    pl_storage_path = (project_root / args.plaintext_storage).resolve() if not Path(args.plaintext_storage).is_absolute() else Path(args.plaintext_storage)
    enc_storage_path = (project_root / enc_storage).resolve() if not Path(enc_storage).is_absolute() else Path(enc_storage)

    storage = {
        'encrypted': {
            'storage_path': str(enc_storage_path),
            'collection_name': enc_collection,
            'storage_size_bytes': _dir_size_bytes(enc_storage_path),
            'points_count': encrypted_rag.retriever.vector_store.get_collection_info().get('points_count'),
        },
        'plaintext': {
            'storage_path': str(pl_storage_path),
            'collection_name': args.plaintext_collection,
            'storage_size_bytes': _dir_size_bytes(pl_storage_path),
            'points_count': plaintext_rag.retriever.vector_store.get_collection_info().get('points_count'),
        },
    }

    # Run
    enc_samples, enc_res = _run_queries(encrypted_rag, queries, 'encrypted', sleep_s=args.sleep_s)
    pl_samples, pl_res = _run_queries(plaintext_rag, queries, 'plaintext', sleep_s=args.sleep_s)

    df = pd.DataFrame([asdict(s) for s in enc_samples + pl_samples])
    df.to_csv(out_dir / 'samples.csv', index=False, encoding='utf-8')
    with open(out_dir / 'samples.jsonl', 'w', encoding='utf-8') as f:
        for s in enc_samples + pl_samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + '\n')

    enc_df = df[df['system'] == 'encrypted']
    pl_df = df[df['system'] == 'plaintext']

    summary = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'dataset': args.dataset,
        'config': {
            'embedding_model': config['embedding']['model_name'],
            'distance_metric': config['vector_db']['distance_metric'],
            'rag_top_k': 5,
            'rerank_enabled': bool(config.get('rerank', {}).get('enabled', False)),
            'llm_base_url': config['llm']['base_url'],
            'llm_model': (config.get('llm', {}).get('default_model') or config['llm'].get('model_name', 'mistral')),
            'n_queries': len(queries),
            'sleep_s': args.sleep_s,
        },
        'storage': storage,
        'encrypted': {
            'latency': _summarize(enc_df),
            'resources': enc_res,
        },
        'plaintext': {
            'latency': _summarize(pl_df),
            'resources': pl_res,
        },
    }

    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md = _md_report(summary)
    (out_dir / 'REPORT.md').write_text(md, encoding='utf-8')

    # also write a stable latest shortcut
    latest = (project_root / 'unencrypted' / 'results' / 'LATEST').resolve()
    latest.write_text(run_id, encoding='utf-8')

    print(f"Wrote report to: {out_dir}")


if __name__ == '__main__':
    main()
