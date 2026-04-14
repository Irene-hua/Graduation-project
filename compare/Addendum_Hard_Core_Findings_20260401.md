# Addendum 硬核结论（带数据）

> 日期：2026-04-01  
> 数据来源：`compare/analysis_addendum.md`（由 `compare/llm_compare_20260401_131033_scored.json` 自动生成）  
> 任务：Single 类型 60 题，模型对比：LLama 2 vs Mistral（RAG 场景）  

---

## 1. 题型分布（用于论文“数据集特征”小节）

基于启发式可复现分组（见 `compare/analysis_addendum.md` 的“题型分组规则”），60 个问题的题型构成如下：

| 题型 | 数量 | 占比 |
|---|---:|---:|
| descriptive（描述类） | 35 | 58.3% |
| time（时间类） | 15 | 25.0% |
| entity（实体类） | 9 | 15.0% |
| other（其他） | 1 | 1.7% |

**可用于论文的解释**：该测试集以描述类问题为主，其次为时间抽取类问题；实体类问题占比相对较小。这意味着模型差异更可能体现在“关键短语覆盖/信息抽取精度/生成策略（扩写 vs 克制）”上，而非单纯的实体识别。

---

## 2. 按题型分组的模型表现（均分 + 胜负，支撑“差异来源”）

### 2.1 描述类（descriptive，n=35）

Mistral 在描述类上呈现一致优势（维度为 0~5 分）：

| 维度 | llama2 | mistral | 差值（mistral-llama2） |
|---|---:|---:|---:|
| correctness | 1.80 | 2.14 | +0.34 |
| faithfulness | 3.60 | 3.89 | +0.29 |
| completeness | 1.80 | 2.11 | +0.31 |
| hallucination | 3.11 | 3.54 | +0.43 |
| fluency | 3.94 | 3.97 | +0.03 |

胜负统计（按单题五维综合得分）：
- llama2：5
- mistral：13
- tie：17

**归因要点（论文可用）**：描述类问题通常要求覆盖 Ground Truth 的关键短语组合。Mistral 更可能覆盖 gold 中的关键信息点，同时在“扩写导致的幻觉风险”上控制更好，因此在 correctness/completeness/hallucination 上形成叠加优势；而两者 fluency 几乎一致。

---

### 2.2 时间类（time，n=15）

时间类上出现“精确抽取能力差异”：Mistral 在 correctness/completeness 上反而低于 llama2，但在 faithfulness/hallucination 上略高。

| 维度 | llama2 | mistral | 差值（mistral-llama2） |
|---|---:|---:|---:|
| correctness | 2.07 | 1.53 | -0.53 |
| faithfulness | 3.67 | 3.73 | +0.07 |
| completeness | 2.07 | 1.53 | -0.53 |
| hallucination | 3.13 | 3.27 | +0.13 |
| fluency | 4.00 | 3.93 | -0.07 |

胜负统计（按单题五维综合得分）：
- llama2：4
- mistral：2
- tie：9

**归因要点（论文可用）**：时间类问题对“时间戳精确对齐”更敏感。Mistral 倾向更保守/克制（faithfulness、hallucination 略高），但在具体时间点抽取上更容易出现偏差或不对齐，导致 correctness/completeness 下滑。若系统核心任务依赖时间线或事件时间戳抽取，建议对 time 类做专项评估/约束输出格式，而不是仅凭总体均分替换模型。

---

### 2.3 实体类（entity，n=9）

实体类问题中两模型整体接近，差异很小：

| 维度 | llama2 | mistral | 差值（mistral-llama2） |
|---|---:|---:|---:|
| correctness | 2.78 | 2.78 | +0.00 |
| faithfulness | 3.78 | 3.78 | +0.00 |
| completeness | 2.78 | 2.78 | +0.00 |
| hallucination | 3.22 | 3.33 | +0.11 |
| fluency | 4.00 | 4.00 | +0.00 |

胜负统计（按单题五维综合得分）：
- llama2：2
- mistral：3
- tie：4

**归因要点（论文可用）**：在 RAG 检索证据较稳定的前提下，实体类更像“从上下文中定位并复述答案”，因此模型差异被明显压缩。

---

## 3. 错误类型占比（Failure Modes，支撑“为什么错”）

基于评分输出中的 debug 信号与分数阈值，进一步将失败模式分解为可解释 error_tag（启发式、可复现）：

### 3.1 llama2 错误类型分布（tag 占比）

| error_tag | count | ratio |
|---|---:|---:|
| ok_or_minor | 25 | 23.6% |
| hallucination_risk | 25 | 23.6% |
| keyword_missing | 24 | 22.6% |
| incorrect | 24 | 22.6% |
| idk | 4 | 3.8% |
| overlong_for_short_gold | 2 | 1.9% |
| time_offset_same_date | 1 | 0.9% |
| hedging | 1 | 0.9% |

### 3.2 mistral 错误类型分布（tag 占比）

| error_tag | count | ratio |
|---|---:|---:|
| ok_or_minor | 23 | 22.3% |
| incorrect | 20 | 19.4% |
| keyword_missing | 20 | 19.4% |
| hallucination_risk | 19 | 18.4% |
| idk | 8 | 7.8% |
| overlong_for_short_gold | 7 | 6.8% |
| hedging | 3 | 2.9% |
| wrong_date | 2 | 1.9% |
| time_offset_same_date | 1 | 1.0% |

**可用于论文的解释**：
- 两模型共同的主要误差来源是 **keyword_missing** 与 **incorrect**，说明 Single 任务下“关键短语/关键事实覆盖与对齐”仍是主要瓶颈。
- Mistral 的 **idk** 占比更高（8 vs 4），体现其更保守的回答策略；该策略有助于降低部分幻觉风险，但在 gold 明确时会造成 completeness 损失。
- Mistral 在部分短答案题上更倾向解释性扩写，表现为 **overlong_for_short_gold** 更高（7 vs 2），在某些场景会提高被判定为“幻觉风险信号”的概率。

---

## 4. RAG 诊断分层：检索稳定性与“检索限制模型能力”证据

从原始 `results/llm_compare_20260401_131033.jsonl` 的 `rag_diagnostics` 汇总得出：

| 指标 | llama2 | mistral |
|---|---:|---:|
| retrieval_empty 次数 | 0 | 0 |
| 平均 num_chunks_retrieved | 15.00 | 15.00 |
| 平均 context_length | 2007.0 | 2007.0 |

**可用于论文的解释**：两模型在本实验中获得的检索证据供给几乎一致（chunks 与 context_length 同质且 retrieval_empty 为 0），因此性能差异主要来自生成阶段的抽取精度与表达策略，而非检索阶段偏差。

---

## 5. 可直接粘贴的论文小结（建议段落）

在 Single 类型 60 题的 RAG 场景对比中，题型以描述类（58.3%）和时间类（25.0%）为主。分组结果显示，Mistral 在描述类问题上在 correctness（+0.34）、completeness（+0.31）与 hallucination（+0.43）等维度形成稳定优势；而在时间类问题上，Mistral 的 faithfulness 与 hallucination 略高，但 correctness 与 completeness 反而低于 LLama 2（均为 -0.53），提示其在时间戳精确对齐方面存在更高风险。错误类型分析进一步表明，两模型的主要失败模式集中在 keyword_missing 与 incorrect；同时 Mistral 的 idk 占比更高（7.8% vs 3.8%）且 overlong_for_short_gold 更高（6.8% vs 1.9%），反映其更保守但更易扩写的生成策略。RAG 诊断指标显示两模型的检索供给一致（retrieval_empty=0，平均 chunk=15，平均 context_length≈2007），因此该对比更能反映生成阶段差异而非检索偏差，并从侧面支持“检索限制模型能力”的现象：当证据充分且同质时，模型间差距被压缩，差异主要体现为对证据的抽取精度与表达策略。

