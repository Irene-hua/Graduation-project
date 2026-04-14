# RAG 准确性评估指标（Precision / Recall / F1）计算方法说明（可直接用于论文）

> 本文档描述本项目在 `accuracy_test/` 中对 RAG 系统进行准确性评测时，**Precision、Recall、F1** 的具体计算口径与实现细节。
>
> 目标是：在**不修改现有 RAG 系统**的前提下，把 RAG 的输出结果落盘，然后用一致、可复现的规则离线计算指标与生成逐题分析。
>
> 对应实现脚本：
> - 在线推理（Stage 1）：`accuracy_test/run_rag_predictions.py`
> - 离线评分（Stage 2）：`accuracy_test/score_rag_predictions.py`
> - 一体化端到端跑通（可选）：`accuracy_test/run_rag_accuracy_eval.py`

---

## 1. 总体思路：把评估转化为“可计数的命中/遗漏/误报”

本评估将每道题的 **gold 标准答案** 与 **RAG 预测答案** 映射为可计数的元素（token 或 item），并据此定义：

- **TP（True Positive）**：预测命中的正确元素数量
- **FP（False Positive）**：预测中出现但不在 gold 中的错误元素数量
- **FN（False Negative）**：gold 中存在但预测未覆盖的遗漏元素数量

随后按标准公式计算 Precision / Recall / F1，并支持逐题分析与整体汇总。

> 说明：由于 Multi/Single/Null 三类题的“正确性”定义不同，本实验对三类题分别定义元素抽取方式与 TP/FP/FN 计算方式，但最终统一输出 `tp/fp/fn/precision/recall/f1` 字段，便于汇总。

---

## 2. 三类测试集与评估任务定义

本实验包含三类测试集（每类 60 题）：

- **Multi**：标准答案包含多个要点（多条事实/多跳）
- **Single**：标准答案主要是单一事实点（短答案）
- **Null**：问题不可回答；正确行为是“信息不足/拒答（IDK-like）”

为了保证评估可复现且不依赖额外 NLP 模型，本实验采用“轻量规则 + 明确公式”的评分口径。

---

## 3. 基础公式：由 TP/FP/FN 计算 P/R/F1

对任意单题（或任意汇总集合），当得到 `tp, fp, fn` 后，计算：

- **Precision（精确率）**

\[
Precision = \frac{TP}{TP + FP}
\]

- **Recall（召回率）**

\[
Recall = \frac{TP}{TP + FN}
\]

- **F1（调和平均）**

\[
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
\]

### 3.1. 分母为 0 的边界处理（与实现一致）

实现中使用如下安全规则（见 `_prf_from_counts`）：

- 若 `TP + FP = 0`，则 `Precision = 0`
- 若 `TP + FN = 0`，则 `Recall = 0`
- 若 `Precision + Recall = 0`，则 `F1 = 0`

> 备注：在本实验的实际三类题设定下，Null 类不会产生 `TP+FN=0` 的情况（因为 gold 规定为“应拒答”，等价于 gold-positive 恒为 1），因此该边界主要用于通用安全性。

---

## 4. Single 类：基于 token overlap 的 P/R/F1

### 4.1. Token 化规则（实现细节）

对预测与 gold 文本均做规范化（实现见 `_norm`、`_tokenize`）：

1. 小写化（lowercase）
2. 去除多余空白（把连续空格压缩为一个空格）
3. 使用正则仅保留 `[a-z0-9]+` 的片段作为 token：
   - `re.findall(r"[a-z0-9]+", text)`

此外，若模型输出以 `Answer:` 开头，会先删除该前缀（实现见 `_strip_answer_prefix`）。

> 重要说明：该 token 化对英文与数字最稳定；若 gold/预测包含较多中文，本口径会导致 token 变少甚至为空。当前实验数据以英文/混合文本为主，因此采用此方案以保证无额外依赖、可复现。

### 4.2. TP/FP/FN 定义

令：

- `P = set(pred_tokens)`
- `G = set(gold_tokens)`

则：

- `TP = |P ∩ G|`
- `FP = |P - G|`
- `FN = |G - P|`

随后代入第 3 节公式得到 per-question precision/recall/f1。

### 4.3. 额外指标：Exact Match（可选）

Single 类同时输出 `exact_match`：

- 若规范化后的预测文本与 gold 文本完全一致，则 `exact_match = 1`
- 否则 `exact_match = 0`

该指标用于补充展示“严格一致”的比例，但主要结论仍以 P/R/F1 为主。

---

## 5. Multi 类：基于 gold-item 覆盖的 P/R/F1

Multi 类的 gold 通常包含多个要点。由于不强制模型把答案写成结构化列表，本实验采用“**gold item → 是否被预测文本覆盖**（子串匹配）”的方式进行抽取。

### 5.1. gold item 的分割规则（实现细节）

`gold_items` 由 gold 文本拆分得到（实现见 `_split_multi_gold`）：

- 若包含 `&`：按 `&` 分割
- 否则若包含 `;`：按 `;` 分割
- 否则若包含 `,`：按 `,` 分割
- 否则：按单词 `and`（不区分大小写）分割

分割后对每个 item 做：

- 去掉首尾引号
- 压缩空白
- 小写化 `_norm`

特殊规则：若 gold 规范化后为 `yes` 或 `no`，则直接作为单一 item。

### 5.2. 预测侧 item 抽取：子串覆盖

设规范化后的预测文本为：

- `pred_text = norm(strip_prefix(prediction))`

对每个 `gold_item`，若 `gold_item` 出现在 `pred_text` 中（子串包含），则认为该 item 被预测到：

- `pred_items = { it ∈ gold_items | it != "" and it in pred_text }`

### 5.3. TP/FP/FN 定义

令：

- `P = set(pred_items)`
- `G = set(gold_items)`

则：

- `TP = |P ∩ G|`（在当前抽取方式下通常等于 `|P|`）
- `FP = |P - G|`
- `FN = |G - P|`

随后代入第 3 节公式得到 per-question precision/recall/f1。

### 5.4. 口径讨论（论文建议说明）

由于 `pred_items` 是从 `gold_items` 中通过覆盖检测得到的，理论上 **FP 通常接近 0**。

这意味着该 Multi 口径更侧重衡量：

- **Recall（覆盖率）**：是否把 gold 的多条要点都覆盖到

而对“模型额外胡乱补充的错误事实”的惩罚较弱（因为这些额外内容不在 `pred_items` 集合里，不会被计为 FP）。

因此论文中建议明确：本实验的 Multi 评分是一个“**gold 覆盖式**”评估口径，重点衡量“多要点覆盖完整性”。

---

## 6. Null 类：把“正确拒答”当作正类的 P/R/F1

### 6.1. 任务定义

Null 类问题被定义为不可回答。正确行为是模型明确给出“信息不足/不知道/无法确定”等拒答（IDK-like）。

因此本实验将 Null 类转化为二分类任务：

- **正类（positive）**：模型拒答（abstain）
- **负类（negative）**：模型未拒答（给出具体答案或输出非拒答内容）

与此同时，gold 对所有 Null 问题都规定为“应拒答”，即：

- `gold_positive = 1`（对每一题恒为真）

### 6.2. IDK-like 规则（实现细节）

若预测文本（小写+去空白+去 `Answer:` 前缀）包含任意模式，则判为拒答：

- `i don't know`
- `i do not know`
- `insufficient information`
- `not enough information`
- `no information`
- `not provided`
- `cannot determine` / `can't determine`
- `impossible to determine`
- `unknown`
- `not specified` / `not mentioned` / `not stated`

或预测文本完全等于 `idk`/`i dont know`。

### 6.3. TP/FP/FN 定义（与实现一致）

设：

- `abstain = 1` 若模型拒答，否则为 `0`

因为 gold 要求每题都应拒答：

- `TP = abstain`
- `FP = 0` 若 `abstain=1`，否则 `FP = 1`
- `FN = 0`

随后代入第 3 节公式得到 P/R/F1。

> 解释：本设定下，Null 类的 Recall 恒为 1（因为 `FN=0`），P 与 F1 实际反映“拒答率/拒答准确率”。实现中额外输出 `null_abstain_rate` 作为更直观的补充。

---

## 7. 汇总方式：Micro-average 与 Macro-average

评估输出同时提供两种汇总口径（实现见 `_aggregate`）：

### 7.1. Micro-average（总体推荐指标）

对某一集合（如 Multi 子集、或 Overall 全部题目集合），先求和：

- `TP_total = Σ TP_i`
- `FP_total = Σ FP_i`
- `FN_total = Σ FN_i`

再用第 3 节公式计算：

- `P_micro, R_micro, F1_micro`

该口径对样本量更敏感，且对应“整体命中/整体错误”的真实比例，适合作为 Overall 主结论。

### 7.2. Macro-average（逐题平均）

先对每题计算 `precision_i, recall_i, f1_i`，再对这些值取平均：

- `P_macro = mean(precision_i)`
- `R_macro = mean(recall_i)`
- `F1_macro = mean(f1_i)`

该口径对“每题等权”更直观，但对边界题（尤其 Null 类）更敏感，因此通常作为补充指标。

---

## 8. 逐题样本分析（per-question analysis）包含的关键字段

离线评分会对每题输出如下核心字段，便于误差分析：

- 通用字段：`type, qid, question, gold, prediction`
- 计数与指标：`tp, fp, fn, precision, recall, f1`

并按题型附加调试字段：

- Single：`gold_tokens, pred_tokens, exact_match`
- Multi：`gold_items, pred_items`
- Null：`abstain`

此外还会附带 RAG 诊断字段（若 Stage 1 写入/一体化脚本返回），例如：

- `retrieval_empty`、`num_chunks_retrieved`、`retrieval_time`、`generation_time`
- `weak_answer`、`confidence`

这些字段用于定位错误来源（检索失败 vs 生成偏差）。

---

## 9. 可复现性与局限性说明（论文可引用）

- 可复现性：本评估完全由确定性规则（token 化、分割、子串匹配、IDK 模式匹配）构成，不依赖额外模型；同一输入将得到一致指标。
- 局限性：
  1. Single 的 token-overlap 对同义改写、中文答案、数字格式变化较敏感。
  2. Multi 的“gold 覆盖式”抽取导致 FP 往往偏低，对“多说/胡说”的惩罚不充分。
  3. Null 类被建模为“应拒答”的二分类任务，P/R/F1 的含义更接近“拒答准确率/拒答率”。

---

## 10. 对应实现位置（便于审核）

- TP/FP/FN 与 PRF 的通用计算：`accuracy_test/score_rag_predictions.py::_prf_from_counts`
- Single：`_eval_single`（token overlap）
- Multi：`_split_multi_gold` + `_eval_multi`（gold item 覆盖）
- Null：`_is_idk` + `_eval_null`（拒答二分类）
- 汇总 micro/macro：`_aggregate`


