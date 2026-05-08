# 实验一（compare）用实验二评分口径重评估：Mistral F1 对比

## 实验二（accuracy_test）已有结果（mistral）
- Overall micro F1: **0.754491** (P=0.750000, R=0.759036)

## 实验一重评估结果（使用 accuracy_test/score_rag_predictions.py 口径，仅 mistral）
| 子集 | micro P | micro R | micro F1 | TP | FP | FN | 输出目录 |
|---|---:|---:|---:|---:|---:|---:|---|
| Multi | 0.538462 | 0.622222 | 0.577320 | 28 | 24 | 17 | `D:\PycharmProjects\Graduation-project\BAcompare\exp1_rescored_with_exp2_rules_20260508\llm_compare_20260401_154508_Multi` |
| Single | 0.566038 | 0.555556 | 0.560748 | 30 | 23 | 24 | `D:\PycharmProjects\Graduation-project\BAcompare\exp1_rescored_with_exp2_rules_20260508\llm_compare_20260401_131033_Single` |
| Null | 0.200000 | 1.000000 | 0.333333 | 12 | 48 | 0 | `D:\PycharmProjects\Graduation-project\BAcompare\exp1_rescored_with_exp2_rules_20260508\llm_compare_20260401_184038_Null` |
| Overall(180) | 0.424242 | 0.630631 | 0.507246 | 70 | 95 | 41 | `D:\PycharmProjects\Graduation-project\BAcompare\exp1_rescored_with_exp2_rules_20260508\llm_compare_20260401_merged_Overall180` |
