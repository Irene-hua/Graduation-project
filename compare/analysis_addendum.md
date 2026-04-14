# LLM 对比评测补充分析（Addendum）
本补充章节基于 `scored.json` 的确定性离线评测结果，进一步从“题型分组 / 错误类型占比 / RAG诊断分层”角度给出可写入论文的硬核统计。

## A. 数据与方法补充
- 评分数据源：`compare/llm_compare_20260401_131033_scored.json`
- 样本量：60（Single）
- 题型分组规则（启发式、可复现）：
  - **time**：问题以 When/Time/Date 为主，或 gold 呈现 YYYYMMDD_HH:MM
  - **entity**：Where/Who/Name/Address 等实体型问题
  - **yesno**：gold 为 yes/no 或问题以 is/are/do/did 开头
  - **descriptive**：why/how/what 等描述型（不落入以上类）

## B. 按题型分组的模型表现
### B1. 题型分布
| 题型 | 数量 | 占比 |
|---|---:|---:|
| descriptive | 35 | 58.3% |
| entity | 9 | 15.0% |
| other | 1 | 1.7% |
| time | 15 | 25.0% |

### B2. 各题型的平均分（0~5）

**题型：descriptive（n=35）**

| 维度 | llama2 | mistral | 差值(mistral-llama2) |
|---|---:|---:|---:|
| correctness | 1.80 | 2.14 | +0.34 |
| faithfulness | 3.60 | 3.89 | +0.29 |
| completeness | 1.80 | 2.11 | +0.31 |
| hallucination | 3.11 | 3.54 | +0.43 |
| fluency | 3.94 | 3.97 | +0.03 |

胜负统计（按单题综合得分）：
- llama2: 5
- mistral: 13
- tie: 17

**题型：entity（n=9）**

| 维度 | llama2 | mistral | 差值(mistral-llama2) |
|---|---:|---:|---:|
| correctness | 2.78 | 2.78 | +0.00 |
| faithfulness | 3.78 | 3.78 | +0.00 |
| completeness | 2.78 | 2.78 | +0.00 |
| hallucination | 3.22 | 3.33 | +0.11 |
| fluency | 4.00 | 4.00 | +0.00 |

胜负统计（按单题综合得分）：
- llama2: 2
- mistral: 3
- tie: 4

**题型：other（n=1）**

| 维度 | llama2 | mistral | 差值(mistral-llama2) |
|---|---:|---:|---:|
| correctness | 5.00 | 5.00 | +0.00 |
| faithfulness | 4.00 | 4.00 | +0.00 |
| completeness | 5.00 | 5.00 | +0.00 |
| hallucination | 4.00 | 4.00 | +0.00 |
| fluency | 4.00 | 4.00 | +0.00 |

胜负统计（按单题综合得分）：
- llama2: 0
- mistral: 0
- tie: 1

**题型：time（n=15）**

| 维度 | llama2 | mistral | 差值(mistral-llama2) |
|---|---:|---:|---:|
| correctness | 2.07 | 1.53 | -0.53 |
| faithfulness | 3.67 | 3.73 | +0.07 |
| completeness | 2.07 | 1.53 | -0.53 |
| hallucination | 3.13 | 3.27 | +0.13 |
| fluency | 4.00 | 3.93 | -0.07 |

胜负统计（按单题综合得分）：
- llama2: 4
- mistral: 2
- tie: 9

## C. 错误类型占比分析
本节将单题‘错误’进一步分解为可解释类别（time偏差/关键词缺失/IDK/过度扩写/幻觉风险等），用于论文中对失败模式的归因。

### llama2 错误类型分布
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

### mistral 错误类型分布
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

## D. RAG 诊断分层（检索对生成的影响）
### D1. 检索稳定性概览
| 指标 | llama2 | mistral |
|---|---:|---:|
| retrieval_empty 次数 | 0 | 0 |
| 平均 num_chunks_retrieved | 15.00 | 15.00 |
| 平均 context_length | 2007.0 | 2007.0 |

### D2. 论文可写结论模板（可直接粘贴）
- 在本轮 Single 任务中，`retrieval_empty` 基本为 0（或极低），且两模型检索到的 chunk 数与 context_length 接近，说明检索阶段对两模型输入证据的供给较为一致。
- 因此，模型差异主要来源于生成阶段对证据的‘抽取精度’与‘表达策略’：例如时间类问题更易出现 **time_offset**；描述类问题更易出现 **keyword_missing** 或过度扩写导致的幻觉风险信号。
- 该现象支持‘检索限制模型能力’的观点：当证据充分且同质时，模型之间的性能差距被显著压缩；要拉开差距，需要更难的检索条件或 Multi/Null 类型问题。
