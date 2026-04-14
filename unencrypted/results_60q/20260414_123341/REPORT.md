# 加密RAG vs 明文RAG：60题三测试集性能对比报告

- run_id: `20260414_123341`
- timestamp: `2026-04-14T12:33:41`
- encrypted_collection: `encrypted_documents_lihua`
- plaintext_collection: `plaintext_documents_lihua_world`

## 资源/存储概览
| system | storage_path | qdrant_dir_size | points | rss_max | cpu_max |
|---|---|---:|---:|---:|---:|
| encrypted | `D:\PycharmProjects\Graduation-project\qdrant_storage` | 10.7 MiB | 2242 | 572.2 MiB | 5.1 |
| plaintext | `D:\PycharmProjects\Graduation-project\unencrypted\qdrant_storage_plaintext` | 10.1 MiB | 2242 | 202.0 MiB | 97.8 |

## 分测试集延迟统计（median/p95）
| dataset | metric | encrypted median | encrypted p95 | plaintext median | plaintext p95 | ratio(median) |
|---|---|---:|---:|---:|---:|---:|
| Multi | retrieval_time_s | 0.1358 | 0.1358 | 0.2514 | 0.2514 | 0.540x |
| Multi | generation_time_s | 58.6810 | 58.6810 | 3.3055 | 3.3055 | 17.752x |
| Multi | total_time_s | 58.8189 | 58.8189 | 3.5592 | 3.5592 | 16.526x |
| Single | retrieval_time_s | 0.2216 | 0.2216 | 0.1585 | 0.1585 | 1.398x |
| Single | generation_time_s | 74.4925 | 74.4925 | 6.2398 | 6.2398 | 11.938x |
| Single | total_time_s | 74.7167 | 74.7167 | 6.4010 | 6.4010 | 11.673x |
| Null | retrieval_time_s | 0.2256 | 0.2256 | 0.2152 | 0.2152 | 1.049x |
| Null | generation_time_s | 77.6049 | 77.6049 | 2.9147 | 2.9147 | 26.626x |
| Null | total_time_s | 77.8333 | 77.8333 | 3.1329 | 3.1329 | 24.844x |

## 结论写作要点（可直接用于论文的小结结构）
1. **检索阶段**：对比 `retrieval_time_s`，可直接量化“加密payload+解密流程”在 Multi/Single/Null 三类问题下的额外开销。
2. **生成阶段**：对比 `generation_time_s`，通常两者接近，说明加密主要影响检索与上下文构建阶段。
3. **资源与存储**：对比 Qdrant 目录大小与 RSS 峰值，讨论 payload 膨胀与解密过程对内存的影响。

---
本报告由 `unencrypted/bench/run_perf_compare_60q.py` 自动生成。
