# Latest recount note (2026-05-08)

This note records the latest manual recount after edits to per-question files.

## Single (llm_compare_20260401_131033_Single)
Source: `.../llm_compare_20260401_131033_Single/per_question.json`

- TP=30, FP=23, FN=24
- micro P=0.566038, R=0.555556, F1=0.560748

Recount artifact: `.../llm_compare_20260401_131033_Single/_recount_from_per_question.json`

## Overall(180) recomputed from subset summaries
Summed micro counts from:
- Multi summary: `.../llm_compare_20260401_154508_Multi/summary.json`
- Single summary: `.../llm_compare_20260401_131033_Single/summary.json`
- Null summary: `.../llm_compare_20260401_184038_Null/summary.json`

- TP=70, FP=95, FN=41
- micro P=0.424242, R=0.630631, F1=0.507246

Audit file: `overall180_recomputed_from_subset_summaries.json`

