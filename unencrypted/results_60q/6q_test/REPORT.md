# 加密RAG vs 明文RAG：60题三测试集性能对比报告

- run_id: `20260414_121133`
- timestamp: `2026-04-14T12:11:33`
- encrypted_collection: `encrypted_documents_lihua`
- plaintext_collection: `plaintext_documents_lihua_world`

## 资源/存储概览
| system | storage_path | qdrant_dir_size | points | rss_max | cpu_max |
|---|---|---:|---:|---:|---:|
| encrypted | `D:\PycharmProjects\Graduation-project\qdrant_storage` | 10.7 MiB | 2242 | 409.6 MiB | 3.9 |
| plaintext | `D:\PycharmProjects\Graduation-project\unencrypted\qdrant_storage_plaintext` | 10.1 MiB | 2242 | 202.5 MiB | 4.2 |

## 分测试集延迟统计（median/p95）
| dataset | metric | encrypted median | encrypted p95 | plaintext median | plaintext p95 | ratio(median) |
|---|---|---:|---:|---:|---:|---:|
| Multi | retrieval_time_s | 0.1996 | 0.2138 | 0.2658 | 0.2725 | 0.751x |
| Multi | generation_time_s | 80.1835 | 80.6224 | 82.0882 | 90.6764 | 0.977x |
| Multi | total_time_s | 80.3856 | 80.8102 | 82.3570 | 90.9394 | 0.976x |
| Single | retrieval_time_s | 0.4253 | 0.4427 | 0.2626 | 0.3430 | 1.620x |
| Single | generation_time_s | 81.3546 | 84.1894 | 76.9006 | 78.6307 | 1.058x |
| Single | total_time_s | 81.7843 | 84.6015 | 77.1656 | 78.8154 | 1.060x |
| Null | retrieval_time_s | 0.3717 | 0.4535 | 0.5891 | 0.6361 | 0.631x |
| Null | generation_time_s | 87.6700 | 91.3485 | 80.8696 | 89.1106 | 1.084x |
| Null | total_time_s | 88.0438 | 91.8039 | 81.4616 | 89.7488 | 1.081x |

## 结论写作要点（可直接用于论文的小结结构）
1. **检索阶段**：对比 `retrieval_time_s`，可直接量化“加密payload+解密流程”在 Multi/Single/Null 三类问题下的额外开销。
2. **生成阶段**：对比 `generation_time_s`，通常两者接近，说明加密主要影响检索与上下文构建阶段。
3. **资源与存储**：对比 Qdrant 目录大小与 RSS 峰值，讨论 payload 膨胀与解密过程对内存的影响。

---
本报告由 `unencrypted/bench/run_perf_compare_60q.py` 自动生成。
