# RAG 系统准确性评估流程与指标（离线评分报告）

本节给出本项目 RAG 系统准确性评测实验的可复现流程与指标定义。
实验采用两阶段方式：先离线生成每题的 RAG 输出（`predictions.jsonl`），再对保存的结果进行离线评分与统计。

## 1. 实验设置（Experimental Setup）

### 1.1 数据集与划分

本实验使用三类测试集（每类 60 题，共 180 题）：

- **Multi**：`data/test_datasets/lihua-queries1`（gold：`data/gold-answer/lihua-queries1-gold-answer`）
- **Single**：`data/test_datasets/lihua-queries2`（gold：`data/gold-answer/lihua-queries2-gold-answer`）
- **Null**：`data/test_datasets/lihua-queries3`（gold 语义为不可回答，应拒答）

其中 Multi/Single 为“gold 非空”的可回答问题，Null 为“应拒答”的不可回答问题。

### 1.2 推理与评分工具链（完全本地）

- **RAG 推理**：复用项目现有 RAG pipeline，不做任何修改。
- **语义裁判（Judge）**：使用本地 Ollama 部署的 LLM（默认 `llama3.2:3b`）判断预测答案与 gold 是否语义一致。
- **拒答识别（Abstention detection）**：使用确定性规则匹配（例如 `insufficient information`、`does not contain any information` 等）识别模型是否拒答。

本次 run 的关键参数（自动记录于 `run_meta.json`）：

- timestamp: 
- rag_llm_model: unknown
- collection_name: 
- config_path: ``
- top_k: 
- temperature: 
- max_tokens: 
- judge_model (Ollama): llama3.2:3b
- judge_timeout: 30s
- judge_retries: 3

### 1.3 两阶段评估流程

**Stage 1（在线推理）**：逐题调用 RAG，保存每题输出到 `predictions.jsonl`。
**Stage 2（离线评分）**：读取 `predictions.jsonl`，对每题进行拒答判断与语义一致性判断，输出逐题明细和汇总指标。

## 2. 指标定义（Precision / Recall / F1）

### 2.1 TP/FP/FN 定义

将每个问题视作一次预测任务，并定义：

- **TP（True Positive）**：预测为正确的次数
- **FP（False Positive）**：预测给出了具体答案但与 gold 语义不一致的次数（答错）
- **FN（False Negative）**：在应回答（Multi/Single）场景下，模型拒答/信息不足导致未回答的次数（漏答）

### 2.2 Precision / Recall / F1 公式

令 TP、FP、FN 为某个集合（如某一测试集或全体样本）上的计数，则：

\[
Precision = \frac{TP}{TP + FP}\,
Recall = \frac{TP}{TP + FN}\,
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}.
\]\n\n为避免除 0，若分母为 0，则对应指标记为 0（实现中 `_prf_from_counts`）。

## 3. 逐题判定逻辑（Multi / Single / Null）

### 3.1 拒答（Abstain）判定

定义函数 `is_abstain(text)`：若模型输出为空、或包含典型拒答表述（如 `i don't know`、`insufficient information`、`does not contain any information`、`the provided context does not contain ...` 等），则判为拒答。

该判定为确定性规则，保证可复现。

### 3.2 语义一致性（Semantic Match）判定

对 Multi/Single（gold 非空）且非拒答的样本，调用本地 Ollama 模型进行语义裁判。Prompt 固定为：

```text
你是一个严格的答案评估器。判断以下两个答案是否语义一致。只回答“是”或“否”。

标准答案：{gold}
预测答案：{pred}
```

若裁判输出不可解析或调用失败，则回退到一个确定性的字符串匹配规则（`fallback`），并在逐题明细中记录 `judge_method` 与 `judge_raw`。

### 3.3 Multi/Single/Null 的统一计分规则

- **Multi / Single（gold 非空）**：
  - 若 `is_abstain(pred)=True`：计为 FN（未回答）
  - 否则若 `is_semantic_match(gold, pred)=True`：计为 TP（回答正确）
  - 否则：计为 FP（回答错误）

- **Null（应拒答）**：
  - 若 `is_abstain(pred)=True`：计为 TP（正确拒答）
  - 否则：计为 FP（未拒答且给出具体答案）

## 4. 判定流程图式伪代码（可直接写入论文）

### 4.1 Multi/Single（gold 非空）

```text
Input: gold, pred
If is_abstain(pred):
    TP=0, FP=0, FN=1      # 拒答 -> 漏答
Else:
    If is_semantic_match(gold, pred):
        TP=1, FP=0, FN=0  # 语义一致 -> 正确
    Else:
        TP=0, FP=1, FN=0  # 语义不一致 -> 答错
Return TP,FP,FN
```

### 4.2 Null（应拒答）

```text
Input: pred
If is_abstain(pred):
    TP=1, FP=0, FN=0      # 正确拒答
Else:
    TP=0, FP=1, FN=0      # 未拒答
Return TP,FP,FN
```

## 5. 汇总统计方法（Micro / Macro）

本实验同时输出 micro-average 与 macro-average：

- **Micro-average**：先对样本集合求和 TP/FP/FN，再代入公式计算 P/R/F1。
- **Macro-average**：先逐题计算 P/R/F1，再取均值。

论文中建议以 micro-average 作为主要指标，因为它更直接反映总体正确/错误/漏答的比例。

## 6. 实验结果（本次运行）

本次运行的 micro-average 指标如下：

- **Multi**: P=0.0000, R=0.0000, F1=0.0000 (TP=0, FP=0, FN=0, n=0)
- **Single**: P=0.0000, R=0.0000, F1=0.0000 (TP=0, FP=0, FN=0, n=0)
- **Null**: P=0.2000, R=1.0000, F1=0.3333 (TP=12, FP=48, FN=0, n=60)
- **Overall**: P=0.2000, R=1.0000, F1=0.3333 (TP=12, FP=48, FN=0, n=60)

并统计 FP/FN 数量（便于错误类型分析）：

- Multi: FP=0, FN=0
- Single: FP=0, FN=0
- Null: FP=48.0, FN=0
- Overall: FP=48.0, FN=0

## 7. 产物文件与可复现性（Artifacts & Reproducibility）

离线评分阶段会在同目录生成以下文件：

- `per_question.csv` / `per_question.json`：逐题明细（含 TP/FP/FN、P/R/F1、judge_method、诊断字段）
- `summary.json`：各子集与 overall 的汇总指标（micro/macro）
- `report.md`：本报告（论文可粘贴版本）
- `error_samples.json` / `error_samples.md`：错误样本（FP/FN）摘录，用于定性分析

其中 `judge_method` 字段用于保证裁判可审计：
- `abstain`：直接由拒答规则判定
- `llm`：由本地 Ollama 模型裁判
- `fallback`：Ollama 调用失败或输出不可解析时的确定性回退规则

## 8. 局限性与威胁（Limitations / Threats to Validity）

1. **语义裁判偏差**：语义一致性由 LLM 裁判，可能受到裁判模型能力与提示词的影响；虽使用本地模型与固定 prompt 以提升可复现性，但仍可能存在误判。
2. **拒答识别覆盖不完全**：`is_abstain` 采用规则匹配，仍可能漏检/误检一些边缘表述。
3. **二值化评分的粒度**：Multi/Single 采用“正确/错误/拒答”三值、每题 TP/FP/FN 取 0/1 的方式，无法区分部分正确（例如 Multi 只覆盖部分要点）的情况。
4. **数据集代表性**：当前测试集规模为 3×60，结论对更大规模或领域迁移的泛化能力仍需进一步实证。

为降低上述威胁，本实验输出逐题明细与错误样本，支持人工抽查与复核。

## 9. 结果解读与结论（论文写作模板，可按需编辑）

本实验在三个测试子集（Multi/Single/Null）与整体（Overall）上报告 micro-average 的 Precision、Recall 与 F1。从结果上看，不同题型的失误模式存在明显差异：Multi 更容易出现时序/因果关系判断错误，Single 更容易出现语义裁判判定为不一致的情况，Null 则主要反映系统在信息不足场景下的拒答能力。

### 9.1 整体表现（Overall）

Overall 指标为 P=0.2000、R=1.0000、F1=0.3333。其中 Precision 主要受到错误回答（FP）数量影响，Recall 主要受到 Multi/Single 的拒答/信息不足导致的漏答（FN）影响。

### 9.2 分题型对比（Multi vs Single vs Null）

- **Multi**：TP 比例约 0.00%，FP 比例约 0.00%，FN 比例约 0.00%。Multi 问题通常涉及多步事实或时间先后关系，系统更容易在关系判断上给出错误结论（FP），或在检索不足时输出信息不足（FN）。
- **Single**：TP 比例约 0.00%，FP 比例约 0.00%，FN 比例约 0.00%。Single 的 gold 往往是单一事实点；在本实验采用的‘语义裁判’口径下，错误更多表现为给出具体答案但与 gold 语义不一致（FP）。
- **Null**：TP（正确拒答）比例约 20.00%，FP（未拒答）比例约 80.00%。Null 子集用于度量系统在不可回答问题上的拒答能力。

### 9.3 主要错误来源（基于逐题字段与裁判方法分布）

- **Null** 的判定来源（judge_method）占比：not_abstain=80.00%, abstain=20.00%
其中 `abstain` 表示直接命中拒答规则；`llm` 表示由本地 Ollama 语义裁判给出一致/不一致判定；`fallback` 表示 Ollama 调用失败或输出不可解析时使用的确定性回退规则。

### 9.4 典型案例（从错误样本中引用）

- Multi-错误回答（FP）示例：本次运行未抽取到对应样本（可提高 --error_samples 或手动从 per_question.csv 筛选）。
- Multi-拒答/信息不足（FN）示例：本次运行未抽取到对应样本（可提高 --error_samples 或手动从 per_question.csv 筛选）。
- **Null-未拒答（FP）示例**（Q1, judge=not_abstain）
  - Question: What time will Li Hua and Wolfgang meet for breakfast on the morning of the 9th?
  - Gold: Insufficient information
  - Prediction:  I don't know. The context does not provide information about any plans for breakfast on the 9th.
以上案例可作为论文中的定性分析材料，用于说明模型在‘关系/时序判断’与‘拒答策略’上的典型失败模式。

### 9.5 结论小结（可直接用于论文）

综合三类测试集结果可以看出：当前 RAG 系统在可回答问题上存在一定比例的错误回答（FP），同时在部分 Multi 问题上出现了拒答/信息不足导致的漏答（FN）。对于 Null 类问题，系统拒答能力仍有提升空间（尤其是减少‘在上下文不足时仍给出具体答案’的情况）。后续优化方向包括：提升检索召回（降低 Multi 的 FN）、增强关系推理与时间顺序建模（降低 Multi 的 FP）、以及引入更严格的拒答触发阈值（提升 Null 的 TP）。

