# 面向隐私保护的轻量RAG系统设计与开发

本科毕业设计 - Privacy-Preserving Lightweight RAG System

## 项目概述

本项目实现了一个面向本地化部署和隐私保护的轻量级检索增强生成（RAG）系统。系统采用 "Ollama + Qdrant + AES加密" 的技术架构，提供安全、高效的本地问答能力。

### 核心特性

- ✅ **文档处理**：支持多种文档格式（TXT, PDF, DOCX, MD）的解析与智能切块
- ✅ **加密保护**：使用AES-256-GCM加密算法保护文档内容
- ✅ **向量存储**：基于Qdrant本地向量数据库的高效存储与检索
- ✅ **轻量级Embedding**：使用sentence-transformers系列轻量级模型生成向量表示
- ✅ **本地LLM**：通过Ollama部署本地大语言模型
- ✅ **模型量化**：支持4-bit量化以降低资源占用
- ✅ **审计日志**：完整的系统访问和操作日志记录，支持完整性校验
- ✅ **性能评估**：多维度评估系统性能（F1、精确率、召回率、响应时间等）

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    RAG System Architecture               │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐      │
│  │ Document │ ───> │  Chunk   │ ───> │ Encrypt  │      │
│  │  Parser  │      │  Text    │      │   (AES)  │      │
│  └──────────┘      └──────────┘      └──────────┘      │
│                                             │            │
│                                             ▼            │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐      │
│  │ Embedding│ ───> │   Qdrant  │ ───> │ Payload  │      │
│  │  Model   │      │ Vector DB │      │ (cipher/ │      │
│  └──────────┘      └──────────┘      │  nonce)  │      │
│                                             │            │
│                                             ▼            │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐      │
│  │ Retrieve │ ───> │ Decrypt  │ ───> │   LLM    │      │
│  │  Top-K   │      │  Chunks  │      │ (Ollama) │      │
│  └──────────┘      └──────────┘      └──────────┘      │
│                                             │            │
│                                             ▼            │
│                                        ┌──────────┐      │
│                                        │  Answer  │      │
│                                        └──────────┘      │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### 实际执行顺序

当前代码的真实问答链路是：

1. **文档处理与加密**：`src.document_processing.ingest` 将原文切块、生成向量，并把每个 chunk 的密文和元数据写入 Qdrant。
2. **检索**：`Retriever` 先用本地 Embedding 生成查询向量，再从 Qdrant 检索候选块。
3. **解密/明文兜底**：如果检索结果带有 `ciphertext + nonce`，则执行 AES 解密；如果结果已包含明文 `text/plaintext/content`，则直接使用，不再强制解密。
4. **上下文构建**：`RAGSystem` 先把检索到的块拼接成上下文，再交给 LLM 生成答案。
5. **主路径生成**：本地 Ollama LLM 基于上下文生成答案，这是默认主路径。
6. **规则兜底**：只有当 LLM 明确无法给出答案（例如返回 `Not found`）时，才启用 `TemporalComparisonRule` 作为窄范围 fallback，且不会改变主链路的 RAG 流程。
7. **审计日志**：记录查询、模型调用与系统访问元数据，但不记录具体查询内容或答案正文。

> 说明：规则系统现在只作为辅助兜底，不参与主链路；主链路始终保持“检索 → 上下文构建 → LLM 生成”的 RAG 流程。

## 技术栈

### 核心依赖
- **Python 3.8+**
- **Qdrant**: 本地向量数据库
- **Sentence Transformers**: 轻量级Embedding模型
- **Ollama**: 本地LLM部署
- **Cryptography**: AES加密实现
- **PyTorch**: 深度学习框架

### 主要库
- `qdrant-client`: Qdrant客户端
- `sentence-transformers`: 预训练Embedding模型
- `transformers`: HuggingFace模型库
- `bitsandbytes`: 模型量化
- `peft`: 参数高效微调
- `pypdf`: PDF文档解析
- `python-docx`: Word文档解析

## 快速开始

### 1. 环境配置

```powershell
# 进入项目根目录
cd D:\PycharmProjects\Graduation-project

# 创建虚拟环境（如果还没有）
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

### 2. 安装 Ollama

```powershell
# 启动 Ollama 服务（单独打开一个终端）
ollama serve

# 拉取模型（示例）
ollama pull llama2
```

### 3. 文档导入

推荐采用“一个数据集一个 collection”的方式，这样不同数据集不会互相污染，便于版本管理与回滚。导入时请以目录为单位传入 `--input_dir`（例如要导入单文件 `data/single_test1/test1.txt`，请把 `--input_dir` 指向 `data/single_test1`）。

命名建议（示例）：
- `encrypted_documents_test1`
- `encrypted_documents_lihua`
- `encrypted_documents_<dataset_name>`

```powershell
# 方案 A：导入到默认 collection（适合单数据集测试）
python -m src.document_processing.ingest --input_dir data\raw --config config\config.yaml --key_file encryption.key --generate_key --reset_collection

# 方案 B：导入到指定 collection（推荐）
python -m src.document_processing.ingest --input_dir data\single_test1 --config config\config.yaml --key_file encryption.key --collection_name encrypted_documents_test1 --reset_collection --log_file logs\ingest_test1.jsonl

# 方案 C：保留历史数据，追加导入到另一个独立 collection
python -m src.document_processing.ingest --input_dir data\lihua_world --config config\config.yaml --key_file encryption.key --collection_name encrypted_documents_lihua --log_file logs\ingest_lihua.jsonl
```

### 4. 运行 RAG 系统

当前命令行入口已经收拢到 `src/rag_pipeline/rag_system.py`，推荐直接通过模块方式启动：

```powershell
# 单个问题问答，指定要查询的 collection
python -m src.rag_pipeline.rag_system --question "What is the date and time that Leslie Hansen responded to Stephanie Panus' email about Wabash and other parties?" --config config\config.yaml --key_file encryption.key --collection_name encrypted_documents_test1 --exact_extract
```

如果你更习惯调用脚本入口，`scripts` 目录里已经不再保留 `run_rag.py`；现在请统一使用上面的模块入口。

> 说明：`--collection_name` 会覆盖 `config.yaml` 中的默认 collection，因此你可以在不修改配置文件的情况下，随时切换查询不同数据集。

### 5. 批量运行

```powershell
# 批量中文评测，指定 collection
python scripts\run_batch_chinese_prompt.py --collection_name encrypted_documents_lihua --queries_file data\test_datasets\Lihua-World-queries

# 通用批处理，指定 collection
python scripts\run_batch_queries.py --collection_name encrypted_documents_test1 --queries_file data\test_datasets\test_queries.txt
```

### 6. 当前命令行使用方式

#### 运行问答

```powershell
python -m src.rag_pipeline.rag_system --question "Your question" --collection_name encrypted_documents_test1
```

#### 可用参数

- `--config`：配置文件路径，默认 `config/config.yaml`
- `--key_file`：加密密钥文件，默认 `encryption.key`
- `--question`：单条问答问题；不传则进入交互模式
- `--top_k`：检索 chunk 数量，默认 `5`
- `--temperature`：LLM 温度，默认 `0.7`
- `--exact_extract`：用于日期/时间类问题的确定性精确抽取
- `--collection_name`：覆盖配置文件中的 Qdrant collection 名称

#### 交互模式

```powershell
python -m src.rag_pipeline.rag_system --collection_name encrypted_documents_lihua
```

#### 常见组合

```powershell
# 指定 collection + 单题问答
python -m src.rag_pipeline.rag_system --question "What is the date and time..." --collection_name encrypted_documents_test1 --exact_extract

# 指定更大的 top_k
python -m src.rag_pipeline.rag_system --question "Your question" --collection_name encrypted_documents_test1 --top_k 10
```

### 7. 统一命令规范

建议按下面的方式组织：

- 导入：`python -m src.document_processing.ingest --input_dir <文档目录> --collection_name <数据集collection> [--reset_collection]`
- 查询：`python -m src.rag_pipeline.rag_system --question "..." --collection_name <数据集collection>`
- 批量：`python scripts\run_batch_chinese_prompt.py --collection_name <数据集collection>`
- 诊断：`python scripts\inspect_collection.py --collection_name <data set collection>`

规则：
1. 一个数据集对应一个 collection，避免互相污染。
2. 如果是全新导入，建议加 `--reset_collection`。
3. 如果要保留旧数据，不要 reset，而是改用新的 collection 名称。
4. 查询前先用 `inspect_collection.py` 确认 collection 里确实有目标文档。

## 项目结构

下面给出当前仓库的精简且可复现的目录组织（仅列出与开发/运行/评估直接相关的顶层目录与关键文件）。此结构旨在帮助读者快速定位源代码、配置、数据与实验输出。

```
D:/PycharmProjects/Graduation-project/
├── src/                          # 源代码（RAG 核心实现）
│   ├── document_processing/      # 文档解析、切块、导入器（ingest）
│   │   ├── ingest.py             # 文档导入模块（CLI + 库接口）
│   │   ├── document_parser.py    # 各类文件解析（txt/pdf/docx/md）
│   │   └── text_chunker.py       # 文本切块器
│   ├── encryption/               # 加密模块（AES-GCM 封装）
│   │   └── aes_encryption.py
│   ├── embedding/                # Embedding 抽象与模型封装
│   │   └── embedding_model.py
│   ├── retrieval/                # 向量存储与检索逻辑（Qdrant 客户端封装）
│   │   ├── vector_store.py
│   │   └── retriever.py
│   ├── llm/                      # 与本地 LLM（Ollama 等）交互的客户端/适配器
│   │   └── ollama.py
│   ├── rag_pipeline/             # RAG 流程：Context 构建、Prompt、主入口
│   │   └── rag_system.py
│   ├── audit/                    # 审计/日志相关工具
│   └── evaluation/               # 评估指标与基线实现
│       └── metrics.py
├── scripts/                      # 辅助脚本（实验、评估、可视化、环境检查）
│   ├── inspect_collection.py     # 检查 Qdrant collection 的工具（建议保留用于调试）
│   ├── run_batch_queries.py      # 批量运行查询/评估脚本（实验用）
│   └── ...                       # 其它实验/评估脚本（可归档）
├── config/                       # 配置文件
│   └── config.yaml               # 主配置
├── data/                         # 原始与测试数据（不含敏感数据）
│   ├── raw/
│   └── test_datasets/
├── accuracy_test/                # 精确度/拒答评估相关脚本与运行结果
├── unencrypted/                  # 明文对照实验代码与数据（性能评估用）
├── results/                      # 持久化的评测结果/对比（LLM 比较、基准等）
├── logs/                         # 运行与审计日志（保留，不删除）
├── README.md                     # 本文件
└── requirements.txt              # 依赖
```

说明与建议
- `src/` 下为系统运行时与核心实现，删除或移动其中任意文件会影响主流程（不建议删除）。
- `scripts/` 下多为实验/评估/可视化脚本，属于工具性质；为保持仓库整洁，可将长期不使用的脚本移动到 `scripts/archive/` 或 `scripts/backup_removed/`。
- `accuracy_test/` 与 `unencrypted/` 中包含复现实验、评估与对照组代码；这些目录对论文实验可复现性重要，建议保留。

## 主要功能模块

### 1. 文档处理模块
- 支持多格式文档解析（TXT, PDF, DOCX, MD）
- 智能文本切块，支持重叠设置
- 保留文档元数据

### 2. 加密模块
- AES-256-GCM加密算法
- 密钥生成与管理
- 密文存储与解密
- 支持密钥派生（PBKDF2）

### 3. Embedding模块
- 多种轻量级模型支持
  - `all-MiniLM-L6-v2` (默认, 384维)
  - `all-MiniLM-L12-v2` (768维)
  - `paraphrase-multilingual-MiniLM-L12-v2` (多语言)
- 批量编码优化
- 余弦相似度计算

### 4. 检索模块
- Qdrant向量数据库集成
- Top-K相似度检索
- 支持多种距离度量（余弦、欧氏、点积）
- **检索结果支持两种形态**：
  - `ciphertext + nonce`：返回密文块，由 `Retriever` 解密后送入 LLM
  - `text/plaintext/content`：如果 payload 中已带明文，则直接使用，不再重复解密
- 支持可选的本地两阶段检索：`Retrieve → Rerank → Context → Generate`
- **引入 Rerank 的原因**：仅依赖向量相似度容易将“语义相近但证据不够精确”的块排到前面；本地精排模块可以在不改变整体架构的前提下，对候选块重新排序，从而提升上下文命中率与最终答案准确率
- **设计原则**：Rerank 作为可选的本地模块插入在 `Retriever` 与 `ContextBuilder` 之间，遵循 `Retrieve → Rerank → Context → Generate` 的两阶段检索架构；若未启用或出现异常，系统会自动退回到原始检索结果，不影响 `RuleEngine` 的 fallback 行为，也不破坏主流程的模块化设计
- **适用场景**：对时间点、邮件头、问答证据位置较敏感的任务，Rerank 可显著减少“相似但不够精确”的候选块进入上下文，提高生成答案的稳定性与可解释性

### 检索结果 payload 规范

向量库中每个文本块建议使用如下 payload 约定：

- `ciphertext`：AES-GCM 加密后的密文，Base64 字符串
- `nonce`：AES-GCM nonce，Base64 字符串
- `chunk_id`：chunk 在文档中的局部编号
- `source_file`：来源文件名
- `doc_id`：文档编号
- 可选明文字段：`text` / `plaintext` / `content` / `source_text`

代码的读取优先级为：

1. 若同时存在 `ciphertext` 和 `nonce`，先解密得到明文
2. 若不存在密文，但存在 `text/plaintext/content/source_text`，直接使用该明文
3. 若两者都没有，则跳过该条结果，不让整条检索链路失败

这意味着：

- **写入阶段**：优先存密文与元数据
- **检索阶段**：优先取密文，必要时可兼容明文 payload
- **生成阶段**：只消费最终明文上下文
- **解释性输出**：系统会额外返回 `retrieve_k`、`rerank_top_scores`、`context_length`、`weak_answer` 等字段，便于论文实验分析与答辩展示

### 5. LLM模块
- Ollama本地部署支持
- 多模型选择（Llama2, Mistral, Phi等）
- 4-bit量化支持
- 推理性能监控

### 6. RAG流程
- 端到端问答链路
- 上下文构建与管理
- 提示词模板定制
- 批量问答处理

### 7. 审计日志
- 系统访问记录
- 查询日志追踪
- 模型调用监控
- 日志完整性验证（SHA256链式哈希）

### 8. 评估模块
- 检索指标：Precision, Recall, F1, MAP, MRR, NDCG
- 答案质量：Exact Match, Token-level F1
- 性能指标：延迟、吞吐量、资源占用
- 系统对比分析

## 配置说明

主配置文件：`config/config.yaml`

### 关键配置项

```yaml
# 文档处理
document_processing:
  chunk_size: 512          # 切块大小
  chunk_overlap: 50        # 重叠字符数

# 加密
encryption:
  key_size: 256           # 密钥长度（位）
  
# Embedding
embedding:
  model_name: 'sentence-transformers/all-MiniLM-L6-v2'
  
# 向量数据库
vector_db:
  collection_name: 'encrypted_documents'
  storage_path: './qdrant_storage'
  
# LLM
llm:
  model_name: 'llama2'
  base_url: 'http://localhost:11434'
  quantization:
    enabled: true
    bits: 4
    
# 检索
retrieval:
  top_k_values: [3, 5, 10, 15]
  default_top_k: 5
```

## 基准测试

### 运行基准测试

```bash
# 测试不同K值
python scripts/run_benchmark.py \
  --test_queries data/test_datasets/test_queries.txt \
  --benchmark_type k_values \
  --output benchmark_results.json

# 测试不同Embedding模型
python scripts/run_benchmark.py \
  --test_queries data/test_datasets/test_queries.txt \
  --benchmark_type embedding_models \
  --output embedding_benchmark.json

# 完整系统测试
python scripts/run_benchmark.py \
  --test_queries data/test_datasets/test_queries.txt \
  --benchmark_type full \
  --output full_benchmark.json
```

### 评估指标

1. **检索性能**
   - Precision@K: 检索结果中相关文档的比例
   - Recall@K: 相关文档被检索到的比例
   - F1 Score: Precision和Recall的调和平均
   - MAP: 平均精度均值
   - NDCG: 归一化折损累积增益

2. **答案质量**
   - Exact Match: 精确匹配率
   - Token-level F1: 词级别F1分数

3. **性能指标**
   - 检索延迟
   - 生成延迟
   - 总响应时间
   - 内存占用
   - GPU占用（如适用）

4. **系统对比**
   - 量化 vs 非量化
   - 加密 vs 非加密
   - 不同K值对比
   - 不同Embedding模型对比

## 安全性与隐私

### 加密机制
- **算法**: AES-256-GCM（Galois/Counter Mode）
- **密钥管理**: 本地文件存储，支持密钥派生
- **完整性**: GCM模式提供认证加密
- **存储**: 仅存储密文，向量基于明文生成

### 审计功能
- 所有系统访问均被记录
- 查询内容和结果计数追踪
- 模型调用详情记录
- SHA256链式哈希确保日志完整性
- 日志轮转和归档

### 本地部署优势
- 数据不离开本地环境
- 无需依赖云服务
- 完全控制数据流向
- 符合严格的隐私法规要求

## 性能优化

### 模型量化
- 4-bit量化减少75%内存占用
- 支持NF4和FP4量化类型
- QLoRA技术保持性能
- 推理速度提升2-4倍

### 检索优化
- 向量索引加速相似度搜索
- 批量编码提高吞吐量
- 缓存机制减少重复计算

### 系统优化
- 异步处理提高并发能力
- 连接池管理数据库连接
- 日志异步写入
- 资源池复用

## 开发指南

### 添加新的文档格式

在 `src/document_processing/document_parser.py` 中添加新的解析方法。示例中若需要引用路径类型，请直接在代码里 `import Path`（来自 `pathlib`），不要把类型注解单独放在 README 的代码识别区域。

```python
# 在源码中新增解析函数
# 例如：
# def _parse_new_format(self, filepath):
#     # 实现新格式的解析逻辑
#     pass
```

### 集成新的Embedding模型

修改配置文件或直接实例化：

```python
from src.embedding import EmbeddingModel

model = EmbeddingModel('your-model-name')
```

### 自定义提示词模板

修改配置文件中的 `rag.prompt_template`：

```yaml
rag:
  prompt_template: |
    根据以下上下文回答问题。
    
    上下文：
    {context}
    
    问题：{question}
    
    回答：
```

## 故障排除

### Ollama 连接失败
```powershell
# 检查 Ollama 服务是否可访问
Invoke-WebRequest http://localhost:11434/api/tags -UseBasicParsing

# 启动服务（在单独窗口执行）
ollama serve
```

### Qdrant 存储错误
```powershell
# 如果看到“Storage folder ... is already accessed by another instance”的报错：
# 1) 关闭其他正在占用 ./qdrant_storage 的 Python/脚本窗口
# 2) 或者直接重启终端后再运行
# 3) 如果要并发使用，请改用 Qdrant server（不要共享本地目录）

# 备份或清空本地向量库（按需执行）
Rename-Item qdrant_storage qdrant_storage_backup -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path qdrant_storage -Force | Out-Null

# 重新导入文档
python -m src.document_processing.ingest --input_dir data\single_test1 --config config\config.yaml --key_file encryption.key
```

### 本地问答结果不准确
```powershell
# 推荐的单文件验证流程
python -m src.document_processing.ingest --input_dir data\single_test1 --config config\config.yaml --key_file encryption.key
python -m src.rag_pipeline.rag_system --question "What is the date and time that Leslie Hansen responded to Stephanie Panus' email about Wabash and other parties?" --config config\config.yaml --key_file encryption.key --exact_extract
```

## 贡献指南

欢迎提交Issue和Pull Request！

### 开发流程
1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

MIT License

## 致谢

- [Ollama](https://ollama.com/) - 本地LLM部署
- [Qdrant](https://qdrant.tech/) - 向量数据库
- [Sentence Transformers](https://www.sbert.net/) - Embedding模型
- [HuggingFace](https://huggingface.co/) - 模型与工具
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - 量化技术

## 联系方式

- 作者：Irene Hua
- GitHub：[@Irene-hua](https://github.com/Irene-hua)
- 项目链接：[https://github.com/Irene-hua/graduation-design](https://github.com/Irene-hua/graduation-design)

## 实验性 Demo（不接入主链路）

为了便于快速实验，本仓库还包含一个**自包含 demo**：`examples/privacy_rag_demo.py`。

- 使用 **Fernet** 做加密
- 使用 **Qdrant :memory:** 做临时向量库
- 带 `CrossEncoder` rerank

该 demo **不符合本项目主设计（AES-256-GCM + Qdrant 持久化/Server + Retriever 解密）**，因此**不被 `scripts/run_rag.py` 使用**。主链路仍以 `src/rag_pipeline/rag_system.py` 中的 `RAGSystem` 为准。

### 5.5 LLM 主模型对比实验（llama2 vs mistral，论文用）

> 注意：如果你的向量库里存的是密文 payload（ciphertext+nonce），对比脚本必须使用**导入/ingest 时同一份** `encryption.key` 才能解密检索结果。
>
> - 如果你之前已经完成过 `ingest_documents.py`，请用当时生成/使用的 key 文件路径：`--key_file encryption.key`
> - **不要**在已有数据集上随意 `--generate_key`，否则新生成的 key 无法解密旧数据（除非你准备重新 ingest）

```powershell
# 例：使用已有 key（推荐）
python scripts\run_llm_comparison.py --queries_file data\test_datasets\Lihua-World-queries --limit 20 --key_file encryption.key --collection_name encrypted_documents_lihua

# 例：如果你是全新环境、还没 ingest，且准备随后重新 ingest 文档，可以生成一个新 key
python scripts\run_llm_comparison.py --queries_file data\test_datasets\Lihua-World-queries --limit 20 --generate_key --key_file encryption.key
```

对比输出（JSONL）保存在 `results/llm_compare_*.jsonl`，可再用以下脚本按 **>60%** 规则自动选择默认主模型并写回配置：

```powershell
python scripts\select_default_llm.py --input results\llm_compare_*.jsonl --write
```
