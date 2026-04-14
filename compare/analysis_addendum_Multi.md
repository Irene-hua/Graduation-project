# LLM 对比评测补充分析（Multi Addendum）
基于 Multi(60题) 离线0~5多维评分结果，从题型/错误类型/RAG诊断角度补充可用于论文的统计结论。

## A. 题型分布
| 题型 | 数量 | 占比 |
|---|---:|---:|
| descriptive | 6 | 10.0% |
| entity | 5 | 8.3% |
| time | 2 | 3.3% |
| yesno | 47 | 78.3% |

## B. 按题型的平均分与胜负

### 题型：descriptive（n=6）
| 维度 | llama2 | mistral | 差值(mistral-llama2) |
|---|---:|---:|---:|
| correctness | 0.33 | 0.50 | +0.17 |
| faithfulness | 3.33 | 3.50 | +0.17 |
| completeness | 0.33 | 0.50 | +0.17 |
| hallucination | 2.67 | 3.00 | +0.33 |
| fluency | 3.83 | 4.00 | +0.17 |

胜负统计（按单题综合得分）：
- llama2: 0
- mistral: 2
- tie: 4

### 题型：entity（n=5）
| 维度 | llama2 | mistral | 差值(mistral-llama2) |
|---|---:|---:|---:|
| correctness | 3.20 | 3.20 | +0.00 |
| faithfulness | 3.60 | 3.80 | +0.20 |
| completeness | 3.20 | 3.20 | +0.00 |
| hallucination | 3.20 | 3.60 | +0.40 |
| fluency | 3.80 | 4.00 | +0.20 |

胜负统计（按单题综合得分）：
- llama2: 0
- mistral: 2
- tie: 3

### 题型：time（n=2）
| 维度 | llama2 | mistral | 差值(mistral-llama2) |
|---|---:|---:|---:|
| correctness | 0.00 | 0.00 | +0.00 |
| faithfulness | 3.00 | 3.00 | +0.00 |
| completeness | 0.00 | 0.00 | +0.00 |
| hallucination | 2.00 | 2.00 | +0.00 |
| fluency | 4.00 | 4.00 | +0.00 |

胜负统计（按单题综合得分）：
- llama2: 0
- mistral: 0
- tie: 2

### 题型：yesno（n=47）
| 维度 | llama2 | mistral | 差值(mistral-llama2) |
|---|---:|---:|---:|
| correctness | 1.28 | 2.23 | +0.96 |
| faithfulness | 3.91 | 3.62 | -0.30 |
| completeness | 1.28 | 2.23 | +0.96 |
| hallucination | 3.32 | 2.94 | -0.38 |
| fluency | 3.98 | 3.96 | -0.02 |

胜负统计（按单题综合得分）：
- llama2: 16
- mistral: 20
- tie: 11

## C. 错误类型占比（Failure Modes）

### llama2
| error_tag | count | ratio |
|---|---:|---:|
| hallucination_risk | 30 | 28.6% |
| incorrect | 24 | 22.9% |
| idk | 18 | 17.1% |
| yesno_incorrect | 14 | 13.3% |
| ok_or_minor | 12 | 11.4% |
| yesno_unparsed | 3 | 2.9% |
| list_miss | 3 | 2.9% |
| list_partial | 1 | 1.0% |

### mistral
| error_tag | count | ratio |
|---|---:|---:|
| hallucination_risk | 30 | 27.8% |
| incorrect | 26 | 24.1% |
| ok_or_minor | 22 | 20.4% |
| yesno_incorrect | 19 | 17.6% |
| idk | 6 | 5.6% |
| list_partial | 2 | 1.9% |
| list_miss | 2 | 1.9% |
| yesno_unparsed | 1 | 0.9% |

## D. RAG 诊断分层
| 指标 | llama2 | mistral |
|---|---:|---:|
| retrieval_empty 次数 | 0 | 0 |
| 平均 num_chunks_retrieved | 15.00 | 15.00 |
| 平均 context_length | 2007.2 | 2007.2 |

## E. 可直接写入论文的结论模板
- Multi 类型任务中，模型除需要做事实判断（是/否）外，还需要覆盖多要点列表并处理跨时间关系。分组结果通常表现为：在 yesno 题上，模型差异更多来自对证据的对齐与否；在 multi_list 题上，差异更多来自列表覆盖率（list_partial/list_miss）。
- 若 RAG 诊断显示两模型的检索供给（chunks/context_length）一致且 retrieval_empty 近似 0，则可将性能差异主要归因于生成阶段（抽取、推断与表达策略），而非检索阶段偏差。
