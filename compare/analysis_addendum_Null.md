# LLM 对比评测补充分析（Null Addendum）

本补充分析聚焦 Null 场景（证据不足/不可回答），用可复现统计刻画：拒答能力、幻觉风险、错误类型与RAG供给稳定性。

## A. Null 场景核心指标
| 指标 | llama2 | mistral |
|---|---:|---:|
| IDK/拒答率 | 46.7% | 60.0% |
| 含具体细节比例（时间/金额等） | 1.7% | 3.3% |

## B. 平均分（0~5）
| 维度 | llama2 | mistral | 差值(mistral-llama2) |
|---|---:|---:|---:|
| correctness | 2.58 | 4.33 | +1.75 |
| faithfulness | 3.53 | 4.58 | +1.05 |
| completeness | 2.58 | 4.33 | +1.75 |
| hallucination | 3.52 | 4.57 | +1.05 |
| fluency | 3.85 | 3.97 | +0.12 |

## C. 错误类型占比（Failure Modes）

### llama2
| error_tag | count | ratio |
|---|---:|---:|
| non_refusal | 32 | 26.9% |
| hallucination_risk | 29 | 24.4% |
| incorrect_non_refusal | 29 | 24.4% |
| idk | 28 | 23.5% |
| specifics_present | 1 | 0.8% |

### mistral
| error_tag | count | ratio |
|---|---:|---:|
| idk | 36 | 46.2% |
| non_refusal | 24 | 30.8% |
| hallucination_risk | 8 | 10.3% |
| incorrect_non_refusal | 8 | 10.3% |
| specifics_present | 2 | 2.6% |

## D. RAG 诊断分层
| 指标 | llama2 | mistral |
|---|---:|---:|
| retrieval_empty 次数 | 0 | 0 |
| 平均 num_chunks_retrieved | 15.00 | 15.00 |
| 平均 context_length | 2006.9 | 2006.9 |

## E. 可直接写入论文的 Null 场景结论要点
- Null 数据集的 gold 全为 ‘Insufficient information’，因此评价重点从‘答对事实’转移到‘识别不可回答并拒答’。在该设定下，IDK/拒答率越高且‘具体细节（时间/金额）输出比例’越低，代表模型在 RAG 证据不足时更安全。
- 若两模型的检索供给指标（chunks/context_length/retrieval_empty）接近，则差异可主要归因于生成阶段策略：某些模型倾向于补全细节（导致 specifics_present 上升），从而在 Hallucination 与 Faithfulness 维度被惩罚。
