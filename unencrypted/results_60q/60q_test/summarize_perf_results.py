"""Summarize perf comparison results from samples.jsonl.

Reads the output produced by `unencrypted/bench/run_perf_compare_60q.py` and generates:
- summary.json: per-system, per-dataset latency stats + resource/size placeholders
- REPORT.md: paper-ready markdown report

This script is intentionally standalone and does NOT touch the encrypted RAG system.

Usage (PowerShell):
  $env:PYTHONUTF8=1
  python summarize_perf_results.py --run_dir .

"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Stats:
    n: int
    mean: float
    median: float
    p90: float
    p95: float


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    w = pos - lo
    return float(sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w)


def _stats(vals: Iterable[float]) -> Stats:
    v = [float(x) for x in vals if x is not None]
    v.sort()
    n = len(v)
    if n == 0:
        return Stats(n=0, mean=0.0, median=0.0, p90=0.0, p95=0.0)
    mean = sum(v) / n
    median = _quantile(v, 0.5)
    p90 = _quantile(v, 0.9)
    p95 = _quantile(v, 0.95)
    return Stats(n=n, mean=mean, median=median, p90=p90, p95=p95)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _fmt_s(x: float) -> str:
    return f"{x:.4f}"


def _fmt_ratio(num: float, den: float) -> str:
    if den == 0:
        return "n/a"
    return f"{(num/den):.3f}x"


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    systems = sorted({r['system'] for r in rows})
    datasets = ['Multi', 'Single', 'Null']
    metrics = ['retrieval_time_s', 'generation_time_s', 'total_time_s']

    summary: Dict[str, Any] = {
        'systems': systems,
        'datasets': datasets,
        'metrics': metrics,
        'by_system': {},
    }

    for sys_name in systems:
        summary['by_system'][sys_name] = {'datasets': {}}
        for ds in datasets:
            dr = [r for r in rows if r.get('system') == sys_name and r.get('dataset') == ds]
            lat: Dict[str, Any] = {}
            for m in metrics:
                st = _stats((float(x.get(m, 0.0)) for x in dr))
                lat[m] = {
                    'n': st.n,
                    'mean': st.mean,
                    'median': st.median,
                    'p90': st.p90,
                    'p95': st.p95,
                }
            ok_rate = sum(1 for x in dr if x.get('ok')) / len(dr) if dr else 0.0
            summary['by_system'][sys_name]['datasets'][ds] = {
                'ok_rate': ok_rate,
                'latency': lat,
            }

    return summary


def build_report(summary: Dict[str, Any]) -> str:
    systems = summary['systems']
    if 'encrypted' in systems and 'plaintext' in systems:
        enc = 'encrypted'
        pl = 'plaintext'
    else:
        enc = systems[0]
        pl = systems[1] if len(systems) > 1 else systems[0]

    lines: List[str] = []
    lines.append('# 加密机制对RAG性能影响：Multi/Single/Null（60题）对比实验报告')
    lines.append('')
    lines.append('## 1. 实验说明')
    lines.append('- 数据集：Multi/Single/Null 各 60 题（lihua-queries1/2/3）')
    lines.append('- 加密侧：encrypted_documents_lihua（加密payload）')
    lines.append('- 明文侧：plaintext_documents_lihua_world（明文payload，独立Qdrant storage）')
    lines.append('')

    lines.append('## 2. 分测试集延迟统计（median / p95）')
    lines.append('| dataset | metric | encrypted median(s) | encrypted p95(s) | plaintext median(s) | plaintext p95(s) | ratio(median) |')
    lines.append('|---|---|---:|---:|---:|---:|---:|')

    for ds in summary['datasets']:
        for m in summary['metrics']:
            e = summary['by_system'][enc]['datasets'][ds]['latency'][m]
            p = summary['by_system'][pl]['datasets'][ds]['latency'][m]
            lines.append(
                f"| {ds} | {m} | {_fmt_s(e['median'])} | {_fmt_s(e['p95'])} | {_fmt_s(p['median'])} | {_fmt_s(p['p95'])} | {_fmt_ratio(e['median'], p['median'])} |"
            )

    lines.append('')
    lines.append('## 3. OK率（系统是否成功走RAG路径）')
    lines.append('| dataset | encrypted ok_rate | plaintext ok_rate |')
    lines.append('|---|---:|---:|')
    for ds in summary['datasets']:
        eok = summary['by_system'][enc]['datasets'][ds]['ok_rate']
        pok = summary['by_system'][pl]['datasets'][ds]['ok_rate']
        lines.append(f"| {ds} | {eok:.3f} | {pok:.3f} |")

    lines.append('')
    lines.append('## 4. 论文可用结论要点（模板）')
    lines.append('1. **总体端到端延迟**：以 `total_time_s` 的 median/p95 为主对比指标。')
    lines.append('2. **检索阶段开销**：以 `retrieval_time_s` 对比，量化加密payload读取与解密带来的额外成本。')
    lines.append('3. **生成阶段差异**：以 `generation_time_s` 对比。若两者接近，说明加密主要影响检索与上下文构建，而非LLM生成。')
    lines.append('4. **分类别讨论**：对 Multi/Single/Null 分别给出加密开销倍数，说明问题类型对检索路径的影响。')
    lines.append('')
    lines.append('---')
    lines.append('本报告由 `summarize_perf_results.py` 自动生成。')
    return '\n'.join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', type=str, default='.', help='Directory containing samples.jsonl')
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    samples = run_dir / 'samples.jsonl'
    if not samples.exists():
        raise SystemExit(f"samples.jsonl not found: {samples}")

    rows = _load_jsonl(samples)
    summary = build_summary(rows)

    (run_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (run_dir / 'REPORT.md').write_text(build_report(summary), encoding='utf-8')
    print(f"Wrote: {run_dir / 'summary.json'}")
    print(f"Wrote: {run_dir / 'REPORT.md'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

