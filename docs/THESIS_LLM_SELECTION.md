# 论文说明：Ollama + GGUF 量化模型的 RAG 主模型选择（llama2 vs mistral）

> 适用场景：Windows + CPU-only（无GPU）的本地隐私保护 RAG 系统。

## 1. 为什么选择 Ollama 方案

1) **本地优先与隐私保护**：Ollama 在本机运行推理服务（HTTP API），文档与检索结果无需上传到云端，符合“数据不出本地”的隐私诉求。

2) **工程集成成本低**：Ollama 提供稳定的 REST API（`/api/generate`、`/api/chat`），RAG 系统只需实现一个轻量 `BaseLLM.generate(prompt)` 适配层，即可自由切换模型。

3) **天然支持 llama.cpp 推理生态**：Ollama 内部基于 `llama.cpp`，默认使用 GGUF 量化权重，适合 CPU-only 环境。

## 2. 为什么 CPU 环境不能使用 bitsandbytes

- `bitsandbytes` 的 4bit/8bit 量化推理主要依赖 CUDA/GPU（即使在部分平台有实验性CPU支持，也常常无法在 Windows + 无GPU 环境稳定安装与运行）。
- 本项目要求可在通用 Windows 机器上复现，因此**禁止引入 GPU 依赖链**（CUDA、bitsandbytes 等）。

结论：在 CPU-only 环境下，更合适的路线是 **llama.cpp / GGUF**（由 Ollama 统一封装）。

## 3. GGUF 量化的作用（降低资源占用）

GGUF 是 `llama.cpp` 生态常用的模型文件格式，常见特性：

- **低比特量化（如 Q4/Q5）**：显著降低内存占用与推理计算量。
- **CPU 友好**：不依赖 GPU/CUDA，适合在普通 CPU 上运行。
- **工程可落地**：配合 Ollama 的模型管理（pull / list / run），部署简单。

在本项目中：
- **向量检索 + rerank + prompt 构造** 仍由 Python 负责；
- **LLM 推理** 由 Ollama 以 GGUF 量化权重在本机 CPU 上完成。

## 4. 为什么选择 Mistral-7B-Instruct 作为候选新主模型

在同等“可本地运行 + CPU 可用 + Ollama 可直接 pull”的约束下，`mistral` 相比 `llama2` 的常见优势（工程经验维度）：

- 指令跟随能力更强（Instruct 对问答任务更友好）
- 在相同参数量级下，输出更稳定，冗余更少
- 与严格 RAG prompt 更匹配：更容易做到“只基于 context 作答”

因此选择 `mistral` 作为主模型候选，同时保留 `llama2` 作为基线对比。

## 5. 对比实验设计（最小可复现）

### 5.1 设计原则

- 不做复杂 benchmark
- 不引入额外训练 / 微调
- 使用**同一个严格 Prompt**，保证公平对比
- 使用 10~20 个问题，快速验证哪一个更适合作为主链路

### 5.2 统一严格 Prompt

系统对所有模型使用同一模板：

- 必须只依据 Context 作答
- Context 不含答案时必须输出 “I don't know”
- 禁止使用先验知识与幻觉

### 5.3 输出与评判标准

每个问题输出：

```json
{
  "question": "...",
  "llama2_answer": "...",
  "mistral_answer": "...",
  "better_model": "mistral / llama2 / tie",
  "reason": "..."
}
```

优先级：
1) 是否基于 context 回答（避免幻觉）
2) 是否命中正确信息
3) 是否出现明显编造/扩写

## 6. 最终主模型选择逻辑

- 若 `mistral` 在对比问题中胜率 **> 60%**，则设置：

`DEFAULT_LLM = "mistral"`

- 否则保持：

`DEFAULT_LLM = "llama2"`

本项目实现方式：写入 `config/config.yaml` 的 `llm.default_model`，RAG 主链路启动时读取该值。

## 7. RAG 稳定性检查（运行时记录）

每次查询会记录：

- `num_chunks_retrieved`：检索块数量
- `retrieval_empty`：是否检索为空（严重错误）
- `rerank_enabled`：rerank 是否启用
- `context_length`：最终 prompt 中 context 的字符长度

这些字段会随批处理输出到 jsonl 结果中，便于论文中展示“稳定性与可解释性”。

