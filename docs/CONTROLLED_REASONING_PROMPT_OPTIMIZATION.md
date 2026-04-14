# 受控推理（Controlled Reasoning）Prompt 优化说明（RAG）

> 目标：在**不引入幻觉**、不使用外部知识、保持 “I don't know” 拒答机制的前提下，让模型能够对 **context 内可逻辑推出**的信息进行**简单推理**（时间推理 / 多句组合 / 指代消解），从而解决“明明检索到了信息却回答不了”的问题。

---

## 1. 为什么原系统会在“需要简单推理”时回答 *I don't know*

原系统采用的提示词（`src/rag_pipeline/prompts.py::STRICT_RAG_PROMPT`）核心约束是：

- **必须只基于 context 作答**
- **如果答案不在 context 中就回答** `"I don't know"`
- 禁止先验知识与幻觉

这种提示词对“直接摘取式问题（extractive QA）”非常有效，但对以下场景存在天然缺陷：

1. **时间指代**：context 出现 “tomorrow / next day / today”，答案并不是原文中的某个字符串，需要把“相对时间”转换成“绝对时间”。
2. **多句组合**：答案分散在多处 chunk 中，需要组合成一个结论。
3. **指代消解**：例如 “he / she / they / that event” 指向上文内容。

由于严格提示词没有允许推理的“授权”，模型在面临这些问题时会倾向于保守拒答（即使 context 已包含推理所需的全部依据），表现为：

- 检索命中 & rerank 正常（有证据）
- 生成仍输出 `I don't know`

---

## 2. Prompt 如何限制了模型能力

严格提示词本质上把任务定义为：

> 在 context 里找到一个“直接出现”的答案片段

而“推理型答案”通常不是一个直接出现的片段，而是：

> 从 context 中抽取事实 → 做有限推理 → 得到答案

如果不给出明确的推理步骤和允许范围，模型会把“推理”误判成“外部知识/猜测”，从而触发自我审查式拒答。

---

## 3. 如何通过“受控推理”解决问题（实现细节）

本次改造引入新提示词：`src/rag_pipeline/prompts.py::CONTROLLED_REASONING_RAG_PROMPT`，并设为主链路默认 prompt（`src/rag_pipeline/rag_system.py::RAGSystem._default_prompt()`）。

与“允许推理”的提示词不同，这个版本是**强制推理（forced reasoning）**：

- 模型必须输出固定结构：Step 1 / Step 2 / Step 3
- 当答案不在原文中显式出现时，**必须先推理再给出答案**
- 只有在推理后仍无法推出时，才允许输出 `"I don't know"`

### 3.1 受控推理 Prompt 的关键设计

**强制但限制推理：**

- ✅ 允许（ONLY）：
  - 时间推理：today / tomorrow / next day（相对时间 → 绝对日期）
  - 简单逻辑组合：把多个事实拼成结论
  - 多句整合：跨 chunk 信息整合
- ❌ 禁止：
  - 外部知识
  - 猜测 context 未出现的信息

**保留安全机制：**

- 仍强制：无法从 context 逻辑推出时，回答 `"I don't know"`

**显式且强制的“推理步骤输出格式”：**

- First extract facts → Then reason step by step → Finally give final answer

并且增加约束：`You MUST NOT answer directly without reasoning.`

### 3.2 时间问题的 Note（强制提供解释框架）

在 context 前加入 Note：

- `"tomorrow" refers to the next day of the given date`
- `All times should be interpreted relative to the timestamp in the context`

这使得模型可以在 context 提供一个锚点时间（例如消息时间戳）时，把相对时间转换为绝对时间。

---

## 4. 验证机制与测试（工程化保障）

为了确保改造满足：

- ✅ Multi 问题能力提升
- ✅ Single 不下降
- ✅ Null 安全性不破坏

本项目加入了两类测试：

1. **Prompt 合同测试（unit tests，无需 Ollama）**
   - `tests/test_controlled_reasoning_prompt.py`
   - 检查 Prompt 是否包含：
     - “simple reasoning”授权
     - “I don't know”拒答
     - “Do NOT use external knowledge / hallucinate”约束
     - 分步骤推理提示
     - 时间推理 Note

2. **时间推理用例契约（unit test）**
   - `tests/test_time_reasoning_contract.py`
   - 用最小上下文构造 tomorrow 场景，确保 Prompt 具备完成推理所需的指令与注记。

> 说明：是否“真的推理成功”取决于具体 LLM（如 mistral）推理稳定性与温度设置，建议在 CLI/批处理上做集成验证；单元测试保证“系统提示词已给足权限与边界”。

---

## 5. 优化前后效果对比（论文写法建议）

### 5.1 优化前

- Prompt 强约束“只能摘取”，未授权推理
- 对 `tomorrow`、组合事实类问题常输出 `I don't know`
- 优点：Null 问题极安全

### 5.2 优化后

- Prompt 授权有限推理 + 明确步骤
- 可以在证据充分时输出推理后的答案
- 仍保留：无法推出即 `I don't know`

### 5.3 示例（时间推理）

问题：

> What time does Li Hua watch the movie "Overwatch 3"?

期望过程：

1) 抽取事实：
- 基准日期：2026-01-21
- 表达：tomorrow
- 时间：7 PM

2) 推理：
- tomorrow → 2026-01-22

3) 输出：
- 2026-01-22 7 PM

---

## 6. 对 Null 问题的安全性说明

受控推理 Prompt 明确要求：

- **Only use information from the context**
- **If cannot be derived, say "I don't know"**

因此：当 context 缺失关键事实时，模型仍将触发拒答，不会因为“允许推理”而放开自由生成。

---

## 7. 代码改动点（便于论文/答辩引用）

- Prompt：`src/rag_pipeline/prompts.py`
  - 新增：`CONTROLLED_REASONING_RAG_PROMPT`
- 主链路默认 Prompt：`src/rag_pipeline/rag_system.py`
  - `_default_prompt()` 从严格模式切换为受控推理模式
- 测试：
  - `tests/test_controlled_reasoning_prompt.py`
  - `tests/test_time_reasoning_contract.py`

---

## 8. 结论

本次 Prompt 工程改造把任务从“纯摘取”升级为“**证据约束下的有限推理**”，核心收益是：

- **解决**：有证据但需要简单推理的问题被误拒答
- **提升**：Multi 类问题的组合能力
- **保持**：Null 问题的安全性与 RAG 的可信性
