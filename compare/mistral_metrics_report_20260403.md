# Mistral 模型在 Single / Multi / Null 三类数据集上的统计与指标（生成于 2026-04-03）

数据来源：
- `compare/llm_compare_20260401_131033_scored.csv`（Single）
- `compare/llm_compare_20260401_154508_Multi_scored.csv`（Multi）
- `compare/llm_compare_20260401_184038_Null_scored.csv`（Null）

## 1. 统计口径

### 1.1 模型筛选
在 CSV 中查找模型列（`model` / `model_name` / `llm` / `llm_name`），筛选值包含 `mistral`（大小写不敏感）的行。

### 1.2 未回答（Unanswered）判定
若答案文本为空或包含以下关键词，则记为未回答：
`I don't know` / `unknown` / `not found` / `no information` / `not in the provided context` / `cannot find` 等。

### 1.3 答对（Correct）判定
优先使用显式正确性列：`is_correct` / `correct` / `match` / `exact_match`。
若不存在，则使用评分列（`score` 或任一包含 `score` 的列），并以阈值判定：
- `score >= 4.0` 视为答对

## 2. 指标定义（Precision / Recall / F1）

- Single/Multi：TP=Correct，FP=Wrong，FN=Wrong+Unanswered
- Null：正确行为是拒答（Unanswered），因此 TP=Unanswered，FP=Answered，FN=0

## 3. 结果汇总（mistral）

| Task | Total | Correct | Wrong | Unanswered | Precision | Recall | F1 |
|------|-------|---------|-------|------------|-----------|--------|----|
| Single | 60 | 23 | 30 | 7 | 0.4340 | 0.3833 | 0.4071 |
| Multi | 60 | 24 | 30 | 6 | 0.4444 | 0.4000 | 0.4211 |
| Null | 60 | 22 | 8 | 30 | 0.5000 | 1.0000 | 0.6667 |

## 4. 解析信息（用于复现/审计）

- **Single**: model_col=`wide:mistral`, answer_col=`mistral_answer`, correctness=`mistral_correctness>=4.0`
- **Multi**: model_col=`wide:mistral`, answer_col=`mistral_answer`, correctness=`mistral_correctness>=4.0`
- **Null**: model_col=`wide:mistral`, answer_col=`mistral_answer`, correctness=`mistral_correctness>=4.0`
