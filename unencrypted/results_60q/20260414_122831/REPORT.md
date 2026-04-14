# 加密RAG vs 明文RAG：60题三测试集性能对比报告

- run_id: `20260414_122831`
- timestamp: `2026-04-14T12:28:31`
- encrypted_collection: `encrypted_documents_lihua`
- plaintext_collection: `plaintext_documents_lihua_world`

## 资源/存储概览
| system | storage_path | qdrant_dir_size | points | rss_max | cpu_max |
|---|---|---:|---:|---:|---:|
| encrypted | `D:\PycharmProjects\Graduation-project\qdrant_storage` | 10.7 MiB | 2242 | 675.2 MiB | 4.4 |
| plaintext | `D:\PycharmProjects\Graduation-project\unencrypted\qdrant_storage_plaintext` | 10.1 MiB | 2242 | 675.2 MiB | 87.8 |

## 分测试集延迟统计（median/p95）
| dataset | metric | encrypted median | encrypted p95 | plaintext median | plaintext p95 | ratio(median) |
|---|---|---:|---:|---:|---:|---:|
| Multi | retrieval_time_s | 0.1410 | 0.1410 | 0.1691 | 0.1691 | 0.834x |
| Multi | generation_time_s | 66.0668 | 66.0668 | 3.2105 | 3.2105 | 20.578x |
| Multi | total_time_s | 66.2094 | 66.2094 | 3.3819 | 3.3819 | 19.578x |
| Single | retrieval_time_s | 0.1266 | 0.1266 | 0.1667 | 0.1667 | 0.760x |
| Single | generation_time_s | 75.6131 | 75.6131 | 6.2500 | 6.2500 | 12.098x |
| Single | total_time_s | 75.7410 | 75.7410 | 6.4186 | 6.4186 | 11.800x |
| Null | retrieval_time_s | 0.1710 | 0.1710 | 0.1643 | 0.1643 | 1.041x |
| Null | generation_time_s | 79.3716 | 79.3716 | 6.3973 | 6.3973 | 12.407x |
| Null | total_time_s | 79.5450 | 79.5450 | 6.5635 | 6.5635 | 12.119x |

## 结论写作要点（可直接用于论文的小结结构）
1. **检索阶段**：对比 `retrieval_time_s`，可直接量化“加密payload+解密流程”在 Multi/Single/Null 三类问题下的额外开销。
2. **生成阶段**：对比 `generation_time_s`，通常两者接近，说明加密主要影响检索与上下文构建阶段。
3. **资源与存储**：对比 Qdrant 目录大小与 RSS 峰值，讨论 payload 膨胀与解密过程对内存的影响。

---
本报告由 `unencrypted/bench/run_perf_compare_60q.py` 自动生成。
