# 加密RAG vs 明文RAG：60题三测试集性能对比报告

- run_id: `20260414_123957`
- timestamp: `2026-04-14T12:39:57`
- encrypted_collection: `encrypted_documents_lihua`
- plaintext_collection: `plaintext_documents_lihua_world`

## 资源/存储概览
| system | storage_path | qdrant_dir_size | points | rss_max | cpu_max |
|---|---|---:|---:|---:|---:|
| encrypted | `D:\PycharmProjects\Graduation-project\qdrant_storage` | 10.7 MiB | 2242 | 677.2 MiB | 9.1 |
| plaintext | `D:\PycharmProjects\Graduation-project\unencrypted\qdrant_storage_plaintext` | 10.1 MiB | 2242 | 676.9 MiB | 4.4 |

## 分测试集延迟统计（median/p95）
| dataset | metric | encrypted median | encrypted p95 | plaintext median | plaintext p95 | ratio(median) |
|---|---|---:|---:|---:|---:|---:|
| Multi | retrieval_time_s | 0.1605 | 0.1677 | 0.1665 | 0.1798 | 0.964x |
| Multi | generation_time_s | 66.5683 | 73.2244 | 76.9029 | 85.3038 | 0.866x |
| Multi | total_time_s | 66.7309 | 73.3941 | 77.0721 | 85.4870 | 0.866x |
| Single | retrieval_time_s | 0.1752 | 0.1799 | 0.1947 | 0.2118 | 0.900x |
| Single | generation_time_s | 69.3080 | 69.6260 | 73.9209 | 74.4499 | 0.938x |
| Single | total_time_s | 69.4858 | 69.7991 | 74.1179 | 74.6637 | 0.938x |
| Null | retrieval_time_s | 0.4530 | 0.6928 | 0.8083 | 1.1444 | 0.560x |
| Null | generation_time_s | 79.9200 | 82.8462 | 73.8839 | 76.2021 | 1.082x |
| Null | total_time_s | 80.3755 | 83.5417 | 74.6960 | 77.3506 | 1.076x |

## 结论写作要点（可直接用于论文的小结结构）
1. **检索阶段**：对比 `retrieval_time_s`，可直接量化“加密payload+解密流程”在 Multi/Single/Null 三类问题下的额外开销。
2. **生成阶段**：对比 `generation_time_s`，通常两者接近，说明加密主要影响检索与上下文构建阶段。
3. **资源与存储**：对比 Qdrant 目录大小与 RSS 峰值，讨论 payload 膨胀与解密过程对内存的影响。

---
本报告由 `unencrypted/bench/run_perf_compare_60q.py` 自动生成。
