# RAG 系统准确性评估实验（Precision / Recall / F1）

> 本文档描述一个**独立的、可复现**的 RAG（Retrieval-Augmented Generation）系统准确性评估实验流程。
> 实验脚本位于 `accuracy_test/` 目录下，不修改现有 RAG 系统实现，仅调用其公开入口进行批量推理与指标计算。

## 1. 研究目的与问题定义

本实验用于定量评估本项目 RAG 系统在三类问题上的回答能力：

- **Multi（多实体/多要点）**：答案包含多个实体或多个要点，需要模型输出完整集合。
- **Single（单一事实）**：答案为单一事实性短答案（时间、地点、名称、原因等）。
- **Null（不可回答）**：在当前知识库/上下文条件下问题不可回答，系统应当**拒答/表明信息不足**，而不是编造。

评估目标指标：

- 精确率（Precision）
- 召回率（Recall）
- F1 分数（F1-score）

并提供：

- **逐题样本分析**（每个问题的预测、金标准、TP/FP/FN、P/R/F1、诊断信息如 confidence、retrieval_time 等）
- **按类型汇总**（Multi / Single / Null 各 60 题）
- **整体汇总**（合并 180 题，计算总体 micro P/R/F1）

## 2. 数据集与金标准

实验使用 3 个测试集，每类 60 题：

| 类型 | 查询文件 | 金标准答案文件 | 题量 |
|---|---|---:|---:|
| Multi | `data/test_datasets/lihua-queries1` | `data/gold-answer/lihua-queries1-gold-answer` | 60 |
| Single | `data/test_datasets/lihua-queries2` | `data/gold-answer/lihua-queries2-gold-answer` | 60 |
| Null | `data/test_datasets/lihua-queries3` | `data/gold-answer/lihua-queries3-gold-answer` | 60 |

数据组织方式为“逐行对齐”：第 *i* 行问题对应第 *i* 行金标准答案。

## 3. 实验对象与运行环境

- 实验对象：项目现有 `src/rag_pipeline/rag_system.py` 中实现的 RAG 系统（检索 + 可选 rerank + LLM 生成）。
- 推理入口：脚本通过调用 `src.rag_pipeline.rag_system._build_runtime(...)` 构建运行时，并对每个问题调用 `rag.answer_question(...)`。

环境前提（与正常 RAG 运行条件一致）：

1. 向量库集合（Qdrant collection）已完成文档导入且非空；
2. LLM 服务可用（例如 Ollama 在 `config/config.yaml` 指定的地址运行）；
3. 项目依赖安装完成（`requirements.txt`）。

> 运行提示（常见报错排查）
>
> - 如果出现 `embeddings.position_ids | UNEXPECTED`：这是 sentence-transformers 载入时的提示，一般不影响运行，可忽略。
> - 如果出现 `RuntimeError: Vector database collection is empty ... has 0 points`：表示当前使用的 collection 没有导入数据。
>   - 解决方式 A（推荐）：用 `--collection_name` 指定一个**已有且非空**的 collection（报错信息里会列出已有 collections）。
>   - 解决方式 B：执行数据导入脚本，将文档写入目标 collection，例如：
>     - `python -m scripts.ingest_documents --input_dir data\single_test1 --collection_name encrypted_documents`
>   - 仅用于调试的方式：加 `--allow_empty_collection` 强制继续运行（此时检索为空，指标不具有参考意义）。

## 4. 指标设计与可复现计算方式

本实验将问答任务转换为“信息抽取式”的 TP/FP/FN 计数，从而定义 P/R/F1。

### 4.1 Single 类型（token-overlap）

- 金标准为一个目标字符串 `gold`。
- 预测为 RAG 输出 `prediction`（脚本会去掉 `Answer:` 前缀）。
- 将 `prediction` 与 `gold` 规范化（小写、去标点、按空白分词），得到 token 集合：
  - `PredTokens`，`GoldTokens`

计数：

- `TP = |PredTokens ∩ GoldTokens|`
- `FP = |PredTokens - GoldTokens|`
- `FN = |GoldTokens - PredTokens|`

并由此计算：

- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1 = 2PR / (P + R)`

备注：同时计算 `Exact Match`（规范化字符串完全相等）。

### 4.2 Multi 类型（set overlap，面向 gold-item 的提取）

Multi 金标准通常用 `&` 分割多个答案项，例如：

- `LiHua & Chae & Yuriko`

步骤：

1. 解析 gold：按 `&`（或 `;` / `,` / `and`）分割为 `GoldItems` 集合；
2. 预测项提取：为了避免模型输出格式差异造成的不稳定，脚本使用一个**可复现规则**：
   - 若某个 `GoldItem`（规范化后）是 `prediction` 的子串，则认为该项被预测到。
   - 得到 `PredItems`。

计数：

- `TP = |PredItems ∩ GoldItems|`
- `FP = |PredItems - GoldItems|`（该提取方式通常接近 0，但仍保留定义）
- `FN = |GoldItems - PredItems|`

并计算 P/R/F1。

### 4.3 Null 类型（正确拒答 / abstention）

Null 集的金标准全部为 `Insufficient information`，即不可回答。

我们将“正确拒答”视为正类：

- `GoldPositive = 1`
- `PredPositive = 1` 当预测文本包含典型拒答模式（例如 `I don't know`、`Insufficient information`、`not enough information` 等）

计数（逐题）：

- 若拒答：`TP=1, FP=0, FN=0`
- 若未拒答（给出具体内容）：`TP=0, FP=1, FN=0`

因此在 Null 集上：

- Precision = Recall = (正确拒答数 / 题目总数)

### 4.4 Overall（总体指标）

将三类问题合并（共 180 条），对每条样本得到的 TP/FP/FN **直接求和**，再计算 micro Precision / Recall / F1。该方式避免了不同类型之间样本量不均的偏置。

同时输出 macro（逐题 P/R/F1 平均）供参考。

## 5. 实验流程（全过程）

1. **准备环境**：安装依赖、启动 LLM 服务（如 Ollama）、确保向量库 collection 已导入。
2. **执行脚本**：运行 `accuracy_test/run_rag_accuracy_eval.py`。
3. **逐题推理**：脚本按顺序读取三个测试集，分别调用 `rag.answer_question(...)` 得到模型回答与检索诊断。
4. **逐题计分**：对每条样本计算 TP/FP/FN 与 P/R/F1，并记录关键诊断字段以便分析误差来源。
5. **汇总统计**：对 Multi/Single/Null 分别汇总 micro/macro P/R/F1；再对全量 180 题汇总 Overall 指标。
6. **导出结果**：保存 JSONL（原始推理输出）、CSV/JSON（逐题分析）、summary.json（汇总）、report.md（简报）。

## 6. 输出文件与逐题样本分析

每次运行会产生一个时间戳目录：

- `accuracy_test/runs/<timestamp>_<llm>_<collection>/predictions.jsonl`
  - 每行一题，包含 RAG 原始返回（answer、retrieval_time、chunk 数、confidence、等）。
- `accuracy_test/runs/<timestamp>_<llm>_<collection>/per_question.csv`
  - 逐题分析表，可直接用于统计/可视化。
- `accuracy_test/runs/<timestamp>_<llm>_<collection>/per_question.json`
  - 与 CSV 同内容，便于二次分析。
- `accuracy_test/runs/<timestamp>_<llm>_<collection>/summary.json`
  - Multi/Single/Null/Overall 的 micro/macro 指标与 TP/FP/FN 汇总。
- `accuracy_test/runs/<timestamp>_<llm>_<collection>/report.md`
  - 本次运行的简要报告（可放入论文附录/实验日志）。

逐题分析建议关注字段：

- `precision/recall/f1`：每题得分
- `weak_answer`：系统内部是否标记为弱回答
- `retrieval_empty` / `num_chunks_retrieved`：检索是否为空、检索到的 chunk 数
- `retrieval_time` / `generation_time`：性能层面的诊断
- Multi 的 `gold_items/pred_items`：多要点题漏答分析

## 7. 结论撰写模板（运行后填充）

> 以下数值来自 `accuracy_test/runs/20260411_201333_pred_encrypted_documents_lihua/summary.json`（LLM=mistral，collection=encrypted_documents_lihua，top_k=5，temperature=0.2）。

- Multi（60题）：micro P=0.622641509434，R=0.6，F1=0.611111111111；TP=33，FP=20，FN=22；macro P=0.58333，R=0.57778，F1=0.58。
- Single（60题）：micro P=0.716981132075，R=0.716981132075，F1=0.716981132075；TP=38，FP=15，FN=15；macro P=0.66667，R=0.65833，F1=0.66111；Exact Match：未在 summary.json 中直接提供（见逐题 `per_question.*` 以计算）。
- Null（60题）：micro P=0.916666666667，R=1.0，F1=0.95652173913；TP=55，FP=5，FN=0；macro P=0.91667，R=0.91667，F1=0.91667；null abstain rate=0.38333。
- Overall（180题）：micro P=0.759036144578，R=0.773006134969，F1=0.765957446809；TP=126，FP=40，FN=37；macro P=0.72222，R=0.71759，F1=0.71926。

- 分析要点建议（结合本次数值）：

1. Multi：micro Recall（0.6）略低于 Precision（0.62264），说明多要点题仍有一定漏答（FN=22），FP=20 表明误报仍存在。建议尝试提高检索覆盖（例如增大 top_k 或改进检索提示），并在生成 prompt 中明确要求列出所有要点，同时在后处理时做严格规范化匹配以减少漏答与错报。
2. Single：micro P/R 均较中等（P=0.717，R=0.717），说明单事实问题总体可靠但仍有改进空间；FP=15、FN=15。建议对答案字符串做更严格的规范化并考虑软匹配策略。
3. Null：拒答检测总体表现良好（recall=1.0），但仍需关注 FP=5 的未拒答误报情况，建议增强拒答模板或将高置信度生成与拒答策略结合使用。
4. Overall：micro F1=0.76596，系统整体表现稳健，但 Precision/Recall 的权衡提示进一步改进检索以提升召回同时控制误报更有价值。

- 运行元信息：

- run timestamp: 20260411_201333
- llm_name: mistral
- collection_name: encrypted_documents_lihua
- top_k: 5
- temperature: 0.2
- schema_version: predictions.v1
- 产出目录（建议引用以便论文附录）：
  - predictions.jsonl, per_question.csv, per_question.json, summary.json, report.md 位于 `accuracy_test/runs/20260411_201333_pred_encrypted_documents_lihua/`

请注意：Exact Match 指标及逐题示例（用于论文中的错误案例展示）可从 `per_question.csv` 中提取并纳入最终论文表格/附录；如果需要我可以基于该目录再提取并生成一页用于论文的“错误示例表格”（含问题、金标准、预测、诊断字段如 retrieval_time / num_chunks_retrieved / weak_answer）。

## 8. 局限性与可扩展方向

- 本实验采用确定性规则计算 P/R/F1，具有可复现性，但对同义改写（paraphrase）不敏感。
- Multi 评估采用“gold-item 子串匹配”以提高稳定性，但对模型输出的额外错误实体不敏感（FP 可能被低估）。

后续可扩展：

- 引入实体抽取/正则模板，增强对 FP 的检测；
- 引入语义相似度（embedding cosine）作为软匹配；
- 将 retrieval ground truth（若可构造）加入，评估检索层指标（Recall@k、MRR、NDCG）。
