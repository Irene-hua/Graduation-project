# llama2 vs mistral（Single 类型 60 题）对比与主模型选择——论文可用说明

> 评测日期：2026-04-01  \
> 运行环境：Windows + CPU-only + Ollama（GGUF/llama.cpp）  \
> 数据集：LiHua-queries2（Type=Single，60 个问题）

## 1. 实验目的

在不改变 RAG 其它环节（检索、加密存储、rerank、prompt 等）的前提下，仅替换主 LLM，比较 `llama2` 与 `mistral` 在同一上下文条件下的回答质量，选择更适合作为主链路的模型，并保留 `llama2` 用于论文对比。

## 2. 实验数据与可复现文件

- Queries：`data/test_datasets/lihua-queries2`
- Gold answers（简略标准答案）：`data/gold-answer/lihua-queries2-gold-answer`
- 两模型回答（对比输出）：`results/llm_compare_20260401_131033.jsonl`

## 3. 评估方法（自动匹配 + 质性分析）

### 3.1 自动匹配判定（用于大样本快速统计）

由于 gold answer 仅提供“最直接答案”（无解释），判定规则采用“宽松命中”：

- Yes/No：归一化后匹配
- 日期/时间：对 `YYYYMMDD` 与 `YYYYMMDD_HH:MM` 归一化后匹配
- 其它：包含（substring）或 token overlap（阈值可调，推荐 >=0.6）
- 若回答为 `I don't know`/缺证据表述：视为未命中（因为 gold 有明确答案）

### 3.2 质性分析维度（论文写作建议）

由于样本量为 60（Single），仍然不足以覆盖所有问题类型（后续还会评测 Multi/Null），因此除准确率外，还建议从以下维度描述：

1) **指令遵循/幻觉风险**：是否严格基于 context，是否出现超出证据的推断。
2) **回答保守性**：缺证据时是否倾向于 “I don’t know”。（更安全，但会拉低覆盖率）
3) **信息覆盖**：实体/列表题是否遗漏关键项或引入多余项。
4) **工程可用性**：回答是否简洁、稳定、易于在日志与评测中解析。

## 4. 定量结果（Single 60题）

本次对比的两模型回答保存在：`results/llm_compare_20260401_131033.jsonl`，
对应的 gold 自动评分结果保存在：`results/llm_compare_20260401_131033_scored.json`，并生成了论文可读报告：`docs/LLM_Comparison_Report_Single_60Q_20260401.md`。

在 **Single 类型 60 题** 的宽松命中评测（substring / token overlap，阈值 0.6）下，统计结果为：

- llama2：**26/60**（**43.33%**）
- mistral：**26/60**（**43.33%**）
- `I don't know` 比例：llama2=**6.67%**，mistral=**13.33%**

结论：在本轮 Single-60 的定量指标上，**两者准确率打平**。

## 5. 多维度分析与最终主模型建议（论文可用）

虽然 Single-60 的正确数相同，但从工程落地角度仍需综合考虑：

### 5.1 指令遵循与“保守性 / 覆盖率”权衡

- `mistral` 在本轮中 `I don't know` 比例更高（13.33% vs 6.67%），体现了在证据不充分时更“保守”。在 **Null 类型** 问题中这往往是优点（更少幻觉），
  但在 **Single 类型**（gold 明确存在答案）中，这会降低可用覆盖率。
- `llama2` 更倾向于给出直接答案，因此在本轮中 `I don't know` 更少。

### 5.2 幻觉风险（需要在 Null 类型进一步验证）

本轮仅为 Single（多数问题在上下文内有答案），无法充分测到幻觉场景。
建议在后续 **Null 类型** 里重点统计：

- “不该回答时是否仍强答”（hallucination rate）
- “是否稳定输出 I don't know”

以决定最终主链路模型。

### 5.3 可读性与工程可用性

- `mistral` 往往更简洁，便于后处理与日志分析。
- `llama2` 有时会给更长解释，但在 CPU-only 环境下长回答也会带来推理耗时增加。

### 5.4 最终建议（当前阶段）

- **仅依据 Single-60：无法得出“mistral 明显优于 llama2”的结论**（准确率打平）。
- 考虑到 `mistral` 在本轮表现出更高的 `I don't know` 比例（覆盖率更低），在“要尽量回答出来”的场景下，
  **建议当前主链路暂时保持 `llama2`**，并把 **`mistral` 作为候选模型继续在 Multi/Null 上评测**。

> 论文写法建议：
> - Single 类型下两模型结果接近（26/60 vs 26/60）。
> - 后续使用 Multi/Null 进一步区分（Multi 看综合推理能力，Null 看幻觉率）。
> - 在三类评测完成后再做最终主模型定稿。

## 6. 统一评测流程（为 Multi / Null 保持一致性）

为保证论文一致性，建议三类问题（Single/Multi/Null）都使用相同的：

- 向量库/检索配置（top_k、rerank开关、同一 prompt）
- 对比输出格式（jsonl：question、llama2_answer、mistral_answer、diagnostics）
- gold 评测方法（同一套宽松命中规则）

### 6.1 生成两模型对比回答（jsonl）

```powershell
python scripts\run_llm_comparison.py `
  --queries_file <TYPE_QUERIES_FILE> `
  --limit <N> `
  --key_file encryption.key `
  --collection_name encrypted_documents_lihua `
  --top_k 15 `
  --max_tokens 128 `
  --timeout 300
```

如遇 Windows 本地 Qdrant `.lock` 锁问题，可加：`--force_copy_storage`。

### 6.2 用 gold 文件自动评分（输出 scored.json + md 报告）

```powershell
python scripts\score_compare_with_gold_inline.py `
  --compare <RESULT_JSONL> `
  --gold <GOLD_FILE> `
  --queries <QUERIES_FILE>
```

输出：

- `<RESULT_STEM>_scored.json`（逐题对错、命中原因、总体统计）
- `docs/LLM_Comparison_Report_<TYPE>_<NQ>.md`（可直接放论文）

> 注：本项目目前对日期时间采用 `YYYYMMDD_HH:MM` 归一化；Multi/Null 继续沿用即可。
