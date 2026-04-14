# Null 场景（不可回答问题）实验评测章节（可直接写入论文）

> 对应产物：
> - 逐题评分与统计：`compare/llm_compare_20260401_184038_Null_scored.json` / `compare/llm_compare_20260401_184038_Null_scored.csv`
> - 逐题分析报告：`compare/LLM_Comparison_Report_Null_60Q_0to5_20260401.md`
> - 补充统计（Addendum）：`compare/analysis_addendum_Null.md`

## 1. 实验设置（Experiment Setup）

本节评测关注 RAG 系统在**证据不足（Null）**情形下的安全表现。与“可回答问题（Single/Multi）”不同，Null 题目被设计为无法从检索内容中得到唯一或确定答案。此时，模型的最优行为应当是**识别不可回答并拒答（abstention）**，同时避免输出任何未经证据支持的具体细节。

- **数据集**：`data/test_datasets/lihua-queries3`（60题）
- **Ground Truth**：`data/gold-answer/lihua-queries3-gold-answer`（每题均为 *Insufficient information*）
- **模型输出**：`results/llm_compare_20260401_184038.jsonl`（包含 llama2 与 mistral 的回答及 `rag_diagnostics`）
- **对比模型**：LLama2 与 Mistral
- **任务类型**：Null（不可回答）

为保证可复现性，本实验采用离线、确定性的启发式评测脚本对每条回答进行 0–5 分多维打分（不使用 LLM-as-a-judge）。

## 2. 评估指标与打分标准（Metrics）

本文采用五维度评分框架，分值区间为 0–5 分。与常规问答评测不同，Null 任务中的“正确性”定义为是否正确拒答。

- **Correctness（正确性）**：
  - 5 分：明确拒答/表明信息不足（例如 *I don't know / the context does not provide… / insufficient information*）
  - 3 分：未显式说 IDK，但明确指出“未提供/未提及/无法确定”，且未给出具体编造
  - 0 分：给出具体结论或推断性答案，未能拒答
- **Context Faithfulness（上下文一致性）**：是否严格基于可见上下文；Null 场景下，不拒答通常意味着更低的一致性。
- **Completeness（完整性）**：拒答是否清晰、是否解释缺失信息来源（如说明“上下文未提供”）。
- **Hallucination（幻觉程度，5=无幻觉）**：是否输出了时间、金额、编号等具体细节；Null 场景下出现“具体细节”通常被视作高风险幻觉。
- **Fluency（表达质量）**：语言是否通顺易读。

此外，为支持“更硬核”的实验论证，我们统计两类 Null 专属指标（详见表 3）：
1) **IDK/拒答率**：回答被识别为拒答（IDK/信息不足表达）的比例；
2) **具体细节输出比例**：回答中出现时间/金额/ID 等具体信息的比例（Null 场景下代表潜在编造风险）。

## 3. 实验结果（Results）

### 3.1 五维平均分对比

表 1 汇总了 60 道 Null 题在五个维度上的平均得分。

**表 1  Null 场景五维平均得分（0–5）**

| 维度 | llama2 | mistral | Δ（mistral-llama2） |
|---|---:|---:|---:|
| Correctness | 2.33 | 3.00 | +0.67 |
| Faithfulness | 3.38 | 3.78 | +0.40 |
| Completeness | 2.33 | 3.00 | +0.67 |
| Hallucination | 3.37 | 3.77 | +0.40 |
| Fluency | 3.85 | 3.97 | +0.12 |

从表 1 可见，Mistral 在 Null 场景下的 Correctness、Completeness 与 Hallucination 均高于 LLama2，表明其对“不可回答”的识别与拒答更稳定，并且整体幻觉风险更低。

### 3.2 单题胜负统计

为避免平均分掩盖题级差异，我们进一步以单题五维总分进行胜负统计（表 2）。

**表 2  Null 场景单题胜负统计（n=60）**

- llama2 胜出：7 题
- mistral 胜出：23 题
- 平局：30 题

胜负统计与表 1 的均值结论一致：Mistral 在更多题目上表现更优。

### 3.3 Null 专属安全性指标（拒答能力与细节风险）

表 3 给出 Null 任务中更能反映系统安全性的两个指标。

**表 3  Null 场景拒答与细节风险统计**

| 指标 | llama2 | mistral |
|---|---:|---:|
| IDK/拒答率 | 46.7% | 60.0% |
| 含具体细节比例（时间/金额等） | 1.7% | 3.3% |

该结果表明：Mistral 更倾向于采取拒答策略（IDK/拒答率更高），因此在 Correctness 与 Completeness 维度取得优势；与此同时需要注意，Mistral 的“具体细节输出比例”略高于 LLama2，意味着其在少数情况下仍存在输出具体信息的风险。

## 4. 讨论（Discussion）

### 4.1 为什么两模型差距不明显（以及差异来自哪里）

Null 数据的 Ground Truth 全为 *Insufficient information*，属于“上限被任务定义钳住”的评测设置：只要模型稳定拒答，评分会集中在高分段。因此两模型差距往往来自少数关键题：
- 是否在证据不足时仍“强行给结论”（导致 Correctness=0、Hallucination 降低）；
- 是否输出时间、金额等具体细节（即使表述上声称“根据上下文”，仍构成幻觉风险）。

### 4.2 llama2 更好的典型原因

基于逐题分析（见 `LLM_Comparison_Report_Null_60Q_0to5_20260401.md`），llama2 的优势题通常具备以下特征：
- 回答更短、更保守，对不可回答问题更倾向于直接拒答；
- 较少“解释性扩写”，因此减少了引入具体细节的机会。

### 4.3 mistral 更好的典型原因

mistral 的优势题主要来源于：
- 更稳定地识别“上下文未提供/未提及/无法确定”，并以拒答形式回答；
- 拒答表述更清晰，常能补充“缺少哪类信息”的说明，从而提升 Completeness。

### 4.4 RAG 诊断：检索供给是否影响 Null 表现

为排除“检索供给差异”导致的评分偏差，我们对 `rag_diagnostics` 进行了分层统计（表 4）。

**表 4  RAG 供给指标（Null，n=60）**

| 指标 | llama2 | mistral |
|---|---:|---:|
| retrieval_empty 次数 | 0 | 0 |
| 平均 num_chunks_retrieved | 15.00 | 15.00 |
| 平均 context_length | 2006.9 | 2006.9 |

从表 4 可见，两模型的检索供给指标完全一致（或近似一致），因此 Null 场景的差异主要来自**生成阶段策略**：是否选择拒答、是否在拒答之外补全细节。

### 4.5 评测偏差与局限性（Threats to Validity）

需要指出，本实验采用确定性启发式评测，虽有利于复现，但仍存在偏差风险：
1) **拒答识别偏差**：IDK/拒答率依赖模式匹配（例如 *not specified / not mentioned*）。若模型用更隐晦的方式表达信息不足，可能被低估。
2) **“具体细节”与“引用上下文细节”难区分**：脚本将时间/金额等视作风险信号，但若上下文确实包含相关细节（即使问题本身不可回答），该规则可能产生误惩罚。
3) **Null gold 的单一性**：本数据集 Ground Truth 均为同一句式（Insufficient information）。这强化了对“拒答”能力的测量，但无法区分“拒答质量”的细粒度差异（例如拒答时的解释性、对缺失信息的定位能力）。

尽管存在上述限制，在 RAG 系统工程实践中，Null 场景强调“保守输出、避免幻觉”，因此本评测仍能有效刻画两模型在安全性方面的相对差异。

## 5. 结论（Conclusion）

综合表 1–表 4 的结果，Mistral 在 Null 场景下整体表现更优：
- 在五维平均分中，Mistral 在 Correctness、Completeness、Faithfulness 与 Hallucination 上均领先；
- 单题胜负统计中，Mistral 胜出题数显著更多（23 vs 7）。

**模型选择建议**：若系统需要提升“证据不足时的安全拒答能力”，Mistral 更适合作为默认回答模型或在低证据/不可回答检测触发时作为兜底模型。但鉴于其“具体细节输出比例”略高，工程上仍建议结合回答后处理（如要求引用证据、对数字/时间进行一致性约束）进一步降低残余幻觉风险。

