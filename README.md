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

1. **文档处理与加密**：`ingest_documents.py` 将原文切块、生成向量，并把每个 chunk 的密文和元数据写入 Qdrant。
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

推荐采用“一个数据集一个 collection”的方式，这样不同数据集不会互相污染。

命名建议：
- `encrypted_documents_test1`
- `encrypted_documents_lihua`
- `encrypted_documents_<dataset_name>`

```powershell
# 方案 A：导入到默认 collection（适合单数据集测试）
python scripts\ingest_documents.py --input_dir data\raw --config config\config.yaml --key_file encryption.key --generate_key --reset_collection

# 方案 B：导入到指定 collection（推荐）
python scripts\ingest_documents.py --input_dir data\single_test1 --config config\config.yaml --key_file encryption.key --collection_name encrypted_documents_test1 --reset_collection --log_file logs\ingest_test1.jsonl

# 方案 C：保留历史数据，追加导入到另一个独立 collection
python scripts\ingest_documents.py --input_dir data\lihua_world --config config\config.yaml --key_file encryption.key --collection_name encrypted_documents_lihua --log_file logs\ingest_lihua.jsonl
```

### 4. 运行 RAG 系统

```powershell
# 单个问题问答，指定要查询的 collection
python scripts\run_rag.py --question "What is the date and time that Leslie Hansen responded to Stephanie Panus' email about Wabash and other parties?" --config config\config.yaml --key_file encryption.key --collection_name encrypted_documents_test1 --exact_extract
```

> 说明：`--collection_name` 会覆盖 `config.yaml` 中的默认 collection，因此你可以在不修改配置文件的情况下，随时切换查询不同数据集。

### 5. 批量运行

```powershell
# 批量中文评测，指定 collection
python scripts\run_batch_chinese_prompt.py --collection_name encrypted_documents_lihua --queries_file data\test_datasets\Lihua-World-queries

# 通用批处理，指定 collection
python scripts\run_batch_queries.py --collection_name encrypted_documents_test1 --queries_file data\test_datasets\test_queries.txt
```

### 6. 统一命令规范

建议按下面的方式组织：

- 导入：`python scripts\ingest_documents.py --input_dir <文档目录> --collection_name <数据集collection> [--reset_collection]`
- 查询：`python scripts\run_rag.py --question "..." --collection_name <数据集collection>`
- 批量：`python scripts\run_batch_chinese_prompt.py --collection_name <数据集collection>`
- 诊断：`python scripts\inspect_collection.py --collection_name <数据集collection>`

规则：
1. 一个数据集对应一个 collection，避免互相污染。
2. 如果是全新导入，建议加 `--reset_collection`。
3. 如果要保留旧数据，不要 reset，而是改用新的 collection 名称。
4. 查询前先用 `inspect_collection.py` 确认 collection 里确实有目标文档。

## 项目结构

```
graduation-design/
├── src/                          # 源代码
│   ├── document_processing/      # 文档解析与切块
│   │   ├── document_parser.py    # 文档解析器
│   │   └── text_chunker.py       # 文本切块器
│   ├── encryption/               # 加密模块
│   │   └── aes_encryption.py     # AES加解密
│   ├── embedding/                # 向量化模块
│   │   └── embedding_model.py    # Embedding模型封装
│   ├── retrieval/                # 检索模块
│   │   ├── vector_store.py       # 向量数据库
│   │   └── retriever.py          # 检索器
│   ├── llm/                      # 语言模型模块
│   │   ├── ollama_client.py      # Ollama客户端
│   │   └── quantized_model.py    # 量化模型支持
│   ├── rag_pipeline/             # RAG流程
│   │   └── rag_system.py         # RAG系统主类
│   ├── audit/                    # 审计模块
│   │   └── audit_logger.py       # 审计日志
│   └── evaluation/               # 评估模块
│       ├── metrics.py            # 评估指标
│       └── benchmarking.py       # 性能基准测试
├── scripts/                      # 运行脚本
│   ├── ingest_documents.py       # 文档导入
│   ├── run_rag.py                # 运行RAG系统
│   ├── run_benchmark.py          # 运行基准测试
│   ├── validate_setup.py         # 环境检查
│   └── test_retrieve.py          # 检索验证
├── config/                       # 配置文件
│   └── config.yaml               # 主配置文件
├── data/                         # 数据目录
│   ├── raw/                      # 原始文档
│   ├── processed/                # 处理后数据
│   └── test_datasets/            # 测试数据集
├── examples/                     # 示例代码
│   └── example_usage.py          # 使用示例
├── tests/                        # 测试代码
├── requirements.txt              # Python依赖
└── README.md                     # 本文件
```

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
python scripts\ingest_documents.py --input_dir data\single_test1 --config config\config.yaml --key_file encryption.key
```

### 本地问答结果不准确
```powershell
# 推荐的单文件验证流程
python scripts\ingest_documents.py --input_dir data\single_test1 --config config\config.yaml --key_file encryption.key
python scripts\run_rag.py --question "What is the date and time that Leslie Hansen responded to Stephanie Panus' email about Wabash and other parties?" --config config\config.yaml --key_file encryption.key --exact_extract
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
