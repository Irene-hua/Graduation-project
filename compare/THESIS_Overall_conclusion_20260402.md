# 基于不同问题类型的RAG系统性能综合分析（Overall Conclusion）

> 本章节为 **Single / Multi / Null** 三类任务的综合评估与统一结论，可直接复制到论文“实验分析与结论”部分。
>
> 数据来源（均为离线确定性评测，0–5 多维评分）：
> - **Single**：`compare/llm_compare_20260401_131033_scored.json` + `compare/LLM_Comparison_Report_Single_60Q_0to5_20260401.md` + `compare/analysis_addendum.md`
> - **Multi**：`compare/llm_compare_20260401_154508_Multi_scored.json` + `compare/LLM_Comparison_Report_Multi_60Q_0to5_20260401.md` + `compare/analysis_addendum_Multi.md`
> - **Null**：`compare/llm_compare_20260401_184038_Null_scored.json` + `compare/LLM_Comparison_Report_Null_60Q_0to5_20260401.md` + `compare/analysis_addendum_Null.md`
>
> 对比模型：**LLaMA 2** 与 **Mistral-7B-Instruct**。

---

## 1. 三类问题实验说明（Task Setup）

为刻画 RAG 系统在不同难度与不同“可回答性”条件下的行为差异，本文将问题划分为三类：

1) **Single（单事实抽取）**：问题通常对应单一实体/时间/地点/原因等短答案。RAG 的作用主要是提供“直接证据片段”，模型任务更接近信息抽取与复述。

2) **Multi（多跳/多要点整合）**：需要跨多条消息或多个证据片段进行整合，尤其在 **yes/no 判断** 与 **多要点列表覆盖** 上更能暴露模型的证据对齐与信息整合能力。

3) **Null（不可回答/证据不足）**：Ground Truth 统一为 *Insufficient information*，该任务不考查事实记忆，而考查模型在证据不足时能否**正确拒答（abstention）**并避免输出编造细节，是 RAG 安全性与幻觉控制的关键压力测试。

三类任务均采用一致的 0–5 多维评分框架：Correctness、Faithfulness、Completeness、Hallucination、Fluency，并结合胜负统计与分组分析（题型/错误类型/RAG诊断）形成综合结论。

---

## 2. 三类问题总结（Type-wise Summary）

### 2.1 Single：简单问题稳定性与抽取能力

在 Single（n=60）中，两模型总体差距较小，表现接近“证据驱动的信息抽取器”。平均分上两者 Correctness 相近（llama2 2.23 vs mistral 2.18），但 Mistral 在 Faithfulness 与 Hallucination 上更高（3.85 vs 3.68；3.48 vs 3.22），表明其在证据边界上更保守。

从题型分组看（`analysis_addendum.md`），Single 的 **time 类问题（n=15）** 上 llama2 的 Correctness/Completeness 更高（2.07 vs 1.53），提示 llama2 在“精确时间点抽取/格式对齐”方面略占优势；而 descriptive 类问题中 Mistral 的整体得分更稳健。

**结论（Single）**：
- 两模型在“证据命中”条件下都能完成较稳定抽取；
- llama2 对部分时间点类短答案更敏感；
- Mistral 更强调保守与证据一致性，降低轻度扩写带来的风险。

### 2.2 Multi：复杂问题能力与信息整合

在 Multi（n=60）中，Mistral 在 Correctness 与 Completeness 上明显领先（2.22 vs 1.35），并在单题胜负上占优（mistral 26 vs llama2 16）。Multi 的题型高度偏向 yes/no（78.3%），该场景下模型需要对证据做更严格的对齐与判断，而不仅仅是复述。

从 addendum 的错误类型与分组结果看：
- **Mistral** 更倾向用“IDK/不确定”避免强行判断，带来更高的 Faithfulness/更低的编造风险，但也可能牺牲部分 completeness；
- **llama2** 更倾向输出确定结论（减少 IDK），但在证据不充分或跨消息关联时更容易出现 yes/no 判断错误或遗漏子要点。

**结论（Multi）**：
- Mistral 在“多要点覆盖/多证据对齐”上整体更优；
- llama2 的风险主要在“过度确定”导致的误判与幻觉；
- Multi 更能体现模型在 RAG 场景下的真实差异（比 Single 更区分）。

### 2.3 Null：幻觉控制与安全拒答能力

在 Null（n=60）中，Mistral 显著优于 llama2：Correctness 4.33 vs 2.58，Faithfulness 4.58 vs 3.53，Hallucination 4.57 vs 3.52；单题胜出也明显更多（mistral 23 vs llama2 1）。补充统计显示 Mistral 的 **IDK/拒答率更高**（60.0% vs 46.7%），说明其在证据不足时更能执行“拒答”策略。

值得注意的是，两模型检索供给指标一致（平均 chunks=15，context_length≈2007，retrieval_empty=0），因此 Null 的差异几乎完全来自**生成阶段策略**：是否愿意拒答、是否会在缺证据时补全细节。

**结论（Null）**：
- Mistral 更安全：拒答更稳定、幻觉风险更低；
- llama2 更容易在不可回答题上给出“貌似合理但无证据”的细节化答案。

---

## 3. 模型总体对比（Model-level Comparison）

### 3.1 LLaMA 2：优势与劣势

**优势**：
- 在 Single 的部分 time 类问题上更容易命中精确时间点（Correctness/Completeness 略高）；
- 表达更直接，较少使用保守措辞，部分明确型问题上更“干脆”。

**劣势**：
- 在 Multi/Null 场景中更容易“过度确定”，当证据不足或需要跨片段整合时，误判与幻觉风险上升；
- Null 的拒答率更低（46.7%），安全性不如 Mistral。

### 3.2 Mistral-7B-Instruct：优势与劣势

**优势**：
- Multi 中 Correctness/Completeness 明显更高，体现出更好的多证据对齐与多要点覆盖能力；
- Null 中拒答更稳定（IDK率 60%），Faithfulness/Hallucination 显著更优，是更安全的 RAG 生成器；
- 在 Single 中整体更保守，平均 Faithfulness/Hallucination 更高。

**劣势**：
- 在 Single 的 time 类短答案上，偶尔因保守/格式缺失导致扣分；
- 在实际部署中，若任务偏好“必须给出具体结论”，则需要结合置信度阈值/证据引用机制，避免过度拒答影响体验。

---

## 4. 关键发现（Cross-cutting Findings）

### 4.1 为什么两个模型差距在 Single 中不明显

Single 多为“单证据-单答案”抽取任务。在检索稳定（retrieval_empty≈0，chunks/context_length 接近）的前提下，模型主要做证据片段的重述与少量改写，导致能力上限被 RAG 输入证据“压缩”，模型差异不易放大。

### 4.2 为什么在 Multi 与 Null 中差异更明显

- **Multi** 需要跨片段整合、处理冲突/顺序/条件，模型必须在“证据支持”与“推断补全”之间做更精细权衡；
- **Null** 则是对“证据边界意识”的直接测试，模型若缺少拒答策略，就会倾向生成看似合理的细节，从而产生幻觉。

因此，与 Single 相比，Multi/Null 更能体现模型在 RAG 系统中的“真实可用性”：是否能在证据不足时克制、在证据分散时整合。

### 4.3 是否存在 RAG 限制模型能力的现象

存在且显著。三类任务的共同诊断表明：当检索供给稳定且相似（chunks/context_length 接近），差异主要来自生成策略；但当检索未覆盖关键证据时，模型上界直接下降，表现会向“IDK 或幻觉补全”两极分化。

换言之，RAG 系统对模型的作用不仅是“增强知识”，更是通过证据供给与上下文窗口对模型施加约束：
- 证据充足时：模型差异被压缩；
- 证据不足时：模型安全策略成为决定性因素（Null）。

### 4.4 哪类问题最能体现模型差异

综合三类结果，最能放大差异的是：
1) **Null（不可回答）**：直接检验拒答/幻觉控制；
2) **Multi-Yes/No 与多要点 list**：检验证据对齐、覆盖率与跨片段整合；
3) **Single-time**：对时间规范化与精确抽取要求高，能体现解析/格式稳定性差异（本实验中 llama2 略占优势）。

---

## 5. 主模型选择（Final Decision）

### 5.1 推荐结论

**推荐使用：Mistral-7B-Instruct 作为 RAG 主模型。**

### 5.2 选择依据（对应统一维度）

1) **整体正确性（Overall Accuracy）**：
- Single 差距很小；Multi 与 Null 中 Mistral 明显更优；因此跨任务综合更稳。

2) **幻觉控制（Hallucination Control）**：
- Null 中 Mistral 的 Hallucination 平均分显著更高（4.57 vs 3.52），并且拒答率更高（60% vs 46.7%），更符合 RAG “以证据为边界”的原则。

3) **复杂问题能力（Multi 能力）**：
- Multi 的 Correctness/Completeness 明显领先，体现更好的多要点整合。

4) **简单问题稳定性（Single 能力）**：
- 两者总体接近；若业务以时间点抽取为主，可对 Mistral 增加输出格式约束或后处理校验。

5) **RAG 适配性（最重要）**：
- 在检索供给一致的条件下，Mistral 更“听话”（更愿意在缺证据时拒答），安全边界更清晰；
- llama2 更可能调用外部常识补全，从工程风险角度不利。

---

## 6. RAG系统整体评价与反思（RAG System Reflection）

### 6.1 优点

- **检索供给稳定**：三类任务中 `retrieval_empty` 基本为 0，且两模型 `num_chunks_retrieved` 与 `context_length` 高度一致，说明检索阶段对模型提供了较一致、可控的证据输入。
- **可复现评测闭环**：使用离线确定性规则在 0–5 多维度对回答进行可解释评分，并支持按题型/错误类型分层，使论文结论具备“可审计性”。

### 6.2 存在的问题

- **证据上界限制**：Multi/Null 结果显示，当检索未能覆盖关键证据或证据不足时，模型表现高度依赖生成策略（拒答/补全）。这意味着检索质量仍是系统瓶颈之一。
- **输出格式与抽取精度约束不足**：Single-time 类问题暴露出时间格式/时分缺失等问题；仅靠生成模型可能不稳定，需要规范化后处理或结构化抽取。

### 6.3 对模型发挥的影响

- 在检索稳定且证据清晰时（多数 Single），模型能力差距被压缩；
- 在证据不足或需要跨片段整合时（Null/Multi），RAG 的检索覆盖度与证据组织方式会直接决定模型上限。

因此，模型选择应与 RAG 策略协同设计：在保证召回覆盖的同时，通过“证据引用/拒答阈值/结构化抽取”抑制幻觉并提升可控性。

---

## 7. 综合结论（Unified Conclusion）

综合 Single、Multi、Null 三类任务，本研究发现：在证据命中条件下，两模型在简单抽取任务上差距有限；但在多证据整合与不可回答场景中，模型的生成策略（是否保守、是否拒答、是否补全细节）成为核心差异来源。基于 Multi 与 Null 的显著优势以及更强的安全拒答能力，本文推荐 **Mistral-7B-Instruct** 作为当前 RAG 系统的主模型，并建议配合检索覆盖提升与结构化后处理，进一步提高系统在真实业务中的鲁棒性与可解释性。

