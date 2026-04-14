# RAG 系统准确性评测实验（两阶段流程）

本目录用于“单独拎出来”的 RAG 准确性评测实验：**不修改现有 RAG 系统实现**，只通过脚本调用已有 pipeline，并把所有评测产物写入 `accuracy_test/`。

## 1. 评测目标

- 使用三类测试集：**Multi / Single / Null**。
- 对每一类分别计算：Precision、Recall、F1。
- 生成逐题样本分析（每题的预测、gold、TP/FP/FN、P/R/F1、检索诊断）。
- 综合三类测试集计算 Overall 的 Precision、Recall、F1（micro-average）。

## 2. 数据集说明

- `data/test_datasets/lihua-queries1`：60 个问题（Type=Multi）
- `data/gold-answer/lihua-queries1-gold-answer`：对应 gold

- `data/test_datasets/lihua-queries2`：60 个问题（Type=Single）
- `data/gold-answer/lihua-queries2-gold-answer`：对应 gold

- `data/test_datasets/lihua-queries3`：60 个问题（Type=Null）
- `data/gold-answer/lihua-queries3-gold-answer`：对应 gold

## 3. 为什么要拆成两阶段

在 Windows + 本地 Qdrant(local mode) 场景下，长时间运行容易遇到：
- `qdrant_storage/.lock` 被残留锁定 / 文件句柄占用导致无法启动
- 中途中断导致“跑一次就要重跑很久”

因此将流程拆为：

1) **在线推理阶段（Stage 1）**：只负责“跑 RAG → 落盘预测结果”。
2) **离线评分阶段（Stage 2）**：只负责“读文件 → 计算指标 → 生成报告”。

这样可以反复调整评分口径/报告模板，而无需重新跑模型。

## 4. Stage 1：批量生成 predictions.jsonl

脚本：`accuracy_test/run_rag_predictions.py`

运行后在 `accuracy_test/runs/<run_id>/` 下生成：
- `predictions.jsonl`：每题一行 JSON（包含 type/qid/question/gold/answer + 诊断字段）
- `run_meta.json`：运行参数与环境记录
- `checkpoint.json`：断点信息
- `config_patched.yaml`（可选）：当启用本地 Qdrant 隔离存储时写出

### Windows Qdrant 锁规避策略

Stage 1 会把工作区的 `qdrant_storage/` **复制**到本次 run 目录下的 `qdrant_storage_copy/`，并在复制时**忽略 `.lock` 文件**。
然后写一个 `config_patched.yaml`，把 `vector_db.storage_path` 指向这份 copy。

这样即使原始 `qdrant_storage/.lock` 被 Windows 锁住，也能跑推理。

## 5. Stage 2：离线评分 + 逐题分析 + 报告

脚本：`accuracy_test/score_rag_predictions.py`

输入：
- `--predictions accuracy_test/runs/<run_id>/predictions.jsonl`

输出（同目录）：
- `per_question.json` / `per_question.csv`
- `summary.json`
- `report.md`

## 6. 指标口径（可写入论文）

为了可复现、无需额外 NLP 依赖，本实验采用轻量但明确的“抽取/覆盖式评分”口径：

- **Single**：
  - 将预测与 gold 文本进行 token 化（小写、去空白、取 `[a-z0-9]+`）。
  - 基于 token 集合重叠计算 TP/FP/FN → Precision/Recall/F1。

- **Multi**：
  - gold 通过 `&`（或 `;`、`,`、`and`）分割为 item 集合。
  - 预测侧不要求输出结构化列表；只要 gold item 在预测文本中出现（大小写无关）就算覆盖。
  - 基于集合重叠计算 TP/FP/FN → Precision/Recall/F1。

- **Null**：
  - 正确行为定义为“拒答/信息不足”（IDK-like）。
  - 若预测文本包含如 `insufficient information / i don't know / not enough information` 等模式，判为 abstain。
  - 用 abstain 作为正类，计算 TP/FP/FN → Precision/Recall/F1。

- **Overall**：
  - micro-average：把三类测试集所有问题的 TP/FP/FN 求和，再计算 PRF。

## 7. 结论撰写建议

- 结论以 `summary.json` 的 Overall（micro）为主，辅以三类子集结果。
- 逐题分析建议从 `per_question.csv` 中筛选：
  - `f1` 最低的若干题（定位失败模式）
  - `retrieval_empty=true` 或 `num_chunks_retrieved=0`（检索失败导致生成失败）
  - `weak_answer=true`（系统内部的弱回答标记）


