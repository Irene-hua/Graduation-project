# 本地 RAG 系统 LLM 替换与对比（llama2 vs mistral）——论文可用说明

> 评测日期：2026-03-31  \
> 运行环境：Windows + CPU-only + Ollama（GGUF/llama.cpp 推理）  \
> 向量库：Qdrant（本地存储）  \
> 数据集：LiHua-World（20 个查询小规模对比）

## 1. 研究目标与约束

本项目的目标不是进行大规模 benchmark，而是在**不引入 GPU 依赖**的前提下，为现有本地 RAG 系统选择一个更合适的主模型。

约束：

- **CPU-only**（无 CUDA / 无 GPU）
- 禁用：bitsandbytes、CUDA、Transformers 4bit 量化（CPU 端不可行或工程成本过高）
- 必须使用：**Ollama**（内部使用 llama.cpp 推理） + **GGUF 量化模型**

候选模型：

- baseline：`llama2`（保留，用于对比）
- 新模型：`mistral`（候选主模型）

## 2. 为什么选择 Ollama + GGUF（工程与论文解释）

1) **CPU 友好、部署简单**：Ollama 封装了 llama.cpp 推理栈，支持直接拉取量化模型（通常为 GGUF），避免了自行编译与复杂依赖。

2) **本地推理与数据安全**：RAG 的上下文包含用户数据/私有文本，默认本地推理可以避免上下文上传到云端。

3) **无需 bitsandbytes**：bitsandbytes 的 4bit/8bit 量化主要服务于 GPU（CUDA）推理；在 CPU-only 环境使用收益低且常伴随兼容成本。

4) **GGUF 量化的作用**：通过低比特量化降低模型内存占用与计算成本，使 7B 级模型在 CPU 上可运行，提高工程落地可行性。

## 3. 对比实验设计（小规模、论文可复现）

### 3.1 评测原则

- 同一数据集
- 同一检索结果（同一 context）
- 同一严格 Prompt：

> You MUST answer ONLY based on the provided context. If the answer is not in the context, say "I don't know".

### 3.2 Gold Answer（人工简略标准答案）

对于 20 个问题，人工给出简略正确答案（允许语义接近，不要求逐字一致）。

### 3.3 评测输入与输出

- 模型回答文件：`results/llm_compare_20260331_184522.jsonl`
- 自动判定脚本：`scripts/evaluate_llm_compare_with_gold.py`
- 自动评测报告：`docs/LLM_Comparison_Report_184522.md`

评分策略：

- 是/否题：抽取 Yes/No
- 日期题：归一化为 `YYYYMMDD`
- 实体列表/长文本：采用 substring 或 token overlap（>=0.6 视为命中）
- 若回答为 “I don't know / not enough information”等：判为不正确

> 注：该自动评测用于论文的“最小对比验证”，并不替代人工抽样复核。

## 4. 实验结果与分析

### 4.1 定量结果（20 题准确率）

来自 `docs/LLM_Comparison_Report_184522.md`：

- **llama2：8/20（40%）**
- **mistral：10/20（50%）**

在该 20 题小样本上，`mistral` 命中略多。

### 4.2 关键现象（质性分析）

1) **短答案/是非题**：两者通常都能给出 Yes/No，但 `mistral` 更倾向于简洁直接；`llama2` 有时会加较长的解释。

2) **信息缺失时的策略**：`mistral` 更容易返回“上下文不足/我不知道”的谨慎回答；这在严格 Prompt 下是安全的，但会降低“必须作答题”的命中率。

3) **实体列表题**：如果上下文包含多个实体，`mistral` 有时能列得更多，但也可能引入不在 gold 中的额外项；`llama2` 有时只返回其中一个。

4) **时间跨度/复杂计算题**：两者都容易出错（例如第 10 题 “about 27 days” 均未命中）。这类题对检索内容组织与明确时间证据更敏感。

### 4.3 小样本结论的局限

- 20 题样本量较小，结论仅代表该数据集、该 Prompt、该检索配置下的表现。
- 更稳健的做法是扩大到 50~100 题并引入人工抽样复核。

## 5. 最终主模型选择建议（工程落地）

综合：

- **在该 20 题 gold 对比中，`mistral` 的准确率略高（50% vs 40%）**；
- `mistral` 作为 7B instruct 模型，在指令遵循与回答结构上通常更稳；
- 系统部署与资源约束上，二者同样可通过 Ollama 的量化推理运行。

因此建议：

- **主链路默认模型：`mistral`**
- **对比实验模型：`llama2`**（保留不删除，用于论文对比章节）

> 若后续扩大样本后 `mistral` 优势不稳定，可回退默认模型为 `llama2`，并在论文中解释“在更大样本/不同问题类型上表现差异”。

## 6. 可复现命令（Windows PowerShell）

### 6.1 对比实验

```powershell
python scripts\run_llm_comparison.py `
  --queries_file data\test_datasets\Lihua-World-queries `
  --limit 20 `
  --key_file encryption.key `
  --collection_name encrypted_documents_lihua `
  --top_k 15 `
  --max_tokens 128 `
  --timeout 300
```

> 若遇到本地 Qdrant 的 `.lock` 占用（Windows 常见），可启用读取副本以规避锁：

```powershell
python scripts\run_llm_comparison.py `
  --queries_file data\test_datasets\Lihua-World-queries `
  --limit 20 `
  --key_file encryption.key `
  --collection_name encrypted_documents_lihua `
  --top_k 15 `
  --max_tokens 128 `
  --timeout 300 `
  --force_copy_storage
```

### 6.2 Gold 自动评分

```powershell
python scripts\evaluate_llm_compare_with_gold.py `
  --input results\llm_compare_20260331_184522.jsonl `
  --output results\llm_compare_20260331_184522_scored.json `
  --report docs\LLM_Comparison_Report_184522.md
```

---

## 附：本次评测报告

- 自动评测报告：`docs/LLM_Comparison_Report_184522.md`

