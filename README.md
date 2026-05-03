# 面向隐私保护的轻量RAG系统设计与开发

本科毕业设计 - Privacy-Preserving Lightweight RAG System

## 项目概述

本项目实现了一个面向本地化部署和隐私保护的轻量级检索增强生成（RAG）系统。系统采用 **Ollama + Qdrant（本地存储） + AES-256-GCM** 的技术架构，提供安全、高效的本地问答能力。

### 核心特性（与当前代码一致）

- ✅ **文档处理**：支持多种文档格式（TXT, PDF, DOCX, MD）的解析与智能切块（`src/document_processing/*`）
- ✅ **加密保护**：使用 **AES-256-GCM** 加密文档 chunk 内容（`src/encryption/aes_encryption.py`）
- ✅ **向量存储**：基于 **Qdrant 本地存储模式**（`storage_path=./qdrant_storage`）存储与检索（`src/retrieval/vector_store.py`）
- ✅ **轻量级 Embedding**：使用 `sentence-transformers` 生成向量（`src/embedding/embedding_model.py`）
- ✅ **本地 LLM**：通过 Ollama 调用本地模型进行生成（`src/llm/ollama.py`）
- ✅ **可选本地 Rerank**：在“向量检索 → 上下文构建”之间插入本地精排（`src/rag_pipeline/rerank.py`，默认启用见 `config/config.yaml`）
- ✅ **审计日志**：提供带链式哈希的审计记录与完整性校验（`src/utils.py` 中 `AuditLogger`，`config/config.yaml` 中 `audit` 配置）
- ✅ **评估与对比脚本**：包含基准测试、对比实验与精度评估脚本（`scripts/`、`accuracy_test/`、`unencrypted/`）

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    RAG System Architecture               │
├─────────────────────────────────────────────────────────┤
│  ┌──────────┐      ┌──────────┐      ┌──────────┐      │
│  │ Document │ ───> │  Chunk   │ ───> │ Encrypt  │      │
│  │  Parser  │      │  Text    │      │ (AES-GCM)│      │
│  └──────────┘      └──────────┘      └──────────┘      │
│                                             │            │
│                                             ▼            │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐      │
│  │ Embedding│ ───> │   Qdrant  │ ───> │ Payload  │      │
│  │  Model   │      │ Vector DB │      │ (cipher/ │      │
│  └──────────┘      └──────────┘      │  nonce + │      │
│                                             │  metadata) │
│                                             ▼            │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐      │
│  │ Retrieve │ ───> │ (Optional)│ ───>│   LLM    │      │
│  │  Top-K   │      │  Rerank   │      │ (Ollama) │      │
│  └──────────┘      └──────────┘      └──────────┘      │
│                                             │            │
│                                             ▼            │
│                                        ┌──────────┐      │
│                                        │  Answer  │      │
│                                        └──────────┘      │
└─────────────────────────────────────────────────────────┘
```

### 实际执行顺序（以代码为准）

当前代码的真实问答链路是：

1. **文档处理与加密导入**：`python -m src.document_processing.ingest` 解析文档、切块、加密 chunk、生成 embedding，并写入 Qdrant collection。
2. **检索**：`Retriever` 用本地 embedding 生成查询向量，从 Qdrant 检索候选 chunk（`src/retrieval/retriever.py`）。
3. **解密/明文兼容**：如果 payload 含 `ciphertext + nonce` 则解密；若 payload 已含明文 `text/plaintext/content/source_text` 则直接使用（检索侧兼容两种形态）。
4. **(可选) Rerank**：若 `config/config.yaml` 的 `rerank.enabled: true`，会用 `LocalReranker` 对候选 chunk 进行本地重排（`src/rag_pipeline/rerank.py`）。
5. **上下文构建**：`ContextBuilder` 过滤/去重/截断，生成上下文字符串。
6. **LLM 生成**：调用 Ollama 本地模型生成答案（`src/llm/ollama.py`），并统一规范输出为 `Answer: ...` 形式。

> 说明：当前仓库**没有** `TemporalComparisonRule` 这类“规则引擎兜底”类；主流程是标准 RAG（检索/可选精排/上下文/生成）。

## 技术栈

### 核心依赖
- **Python 3.8+**
- **Qdrant**：本地向量数据库（本项目默认走 `storage_path` 本地目录模式）
- **Sentence Transformers**：Embedding 与（可选）CrossEncoder（如果启用 `HybridReranker`）
- **Ollama**：本地 LLM 推理服务
- **Cryptography**：AES-GCM 加密

### 主要库（见 `requirements.txt`）
- `qdrant-client`
- `sentence-transformers`
- `transformers`（用于 embedding/通用模型工具链；主生成推理走 Ollama API）
- `pypdf`、`python-docx`、`beautifulsoup4`
- `pytest`、`scikit-learn`、`psutil` 等

## 快速开始

### 1. 环境配置

```powershell
cd D:\PycharmProjects\Graduation-project
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 启动 Ollama

```powershell
# 单独打开一个终端启动服务
ollama serve

# 示例：拉取模型
ollama pull mistral
```

### 3. 文档导入（写入 Qdrant collection）

导入器入口在 `src/document_processing/ingest.py`：

```powershell
# 导入并重置 collection（推荐用于全新实验）
python -m src.document_processing.ingest --input_dir data\single_test1 --config config\config.yaml --key_file encryption.key --reset_collection --collection_name encrypted_documents_test1

# 如果 encryption.key 不存在，可加 --generate_key 自动生成
python -m src.document_processing.ingest --input_dir data\single_test1 --config config\config.yaml --key_file encryption.key --generate_key --reset_collection --collection_name encrypted_documents_test1

# 追加导入到另一个 collection（不 reset）
python -m src.document_processing.ingest --input_dir data\raw --config config\config.yaml --key_file encryption.key --collection_name encrypted_documents_raw
```

> 注意：`--input_dir` 必须是目录；导入器会递归扫描目录下支持的格式（见 `config/config.yaml` 的 `supported_formats`）。

### 4. 运行 RAG 问答（CLI）

命令行入口在 `src/rag_pipeline/rag_system.py`：

```powershell
# 单题问答
python -m src.rag_pipeline.rag_system --question "Your question" --config config\config.yaml --key_file encryption.key --collection_name encrypted_documents_test1

# 交互模式（不传 --question）
python -m src.rag_pipeline.rag_system --config config\config.yaml --key_file encryption.key --collection_name encrypted_documents_test1
```

#### CLI 可用参数（以代码为准）

- `--config`：配置文件路径，默认 `config/config.yaml`
- `--key_file`：加密密钥文件，默认 `encryption.key`
- `--question`：单条问答问题；不传则进入交互模式
- `--top_k`：用于构建上下文的 chunk 数量（默认 `5`）
- `--temperature`：LLM 温度（默认 `0.2`）
- `--collection_name`：覆盖配置文件中的 Qdrant collection
- `--allow_empty_collection`：允许 collection 为空时运行（仅调试用；默认会报错提示先导入）

> 说明：README 旧版本里出现过 `--exact_extract` 参数；当前代码中**不存在**该参数。

### 5. 运行 Web UI（Streamlit，可选）

仓库根目录包含 `streamlit_app.py`，可用 Streamlit 启动一个简单界面：

```powershell
streamlit run streamlit_app.py
```

（如果你的环境未安装 `streamlit`，需要自行安装；本仓库的 `requirements.txt` 当前未强制包含它，以保持依赖轻量。）

### 6. 批量运行脚本

```powershell
# 批量中文查询
python scripts\run_batch_chinese_prompt.py --collection_name encrypted_documents_test1 --queries_file data\test_datasets\Lihua-World-queries

# 通用批处理
python scripts\run_batch_queries.py --collection_name encrypted_documents_test1 --queries_file data\test_datasets\test_queries.txt

# 检查 collection 状态（用于确认是否已导入、points 数量等）
python scripts\inspect_collection.py --collection_name encrypted_documents_test1
```

## 项目结构（与当前仓库一致）

```
D:/PycharmProjects/Graduation-project/
├── src/                          # RAG 核心实现
│   ├── document_processing/      # 解析/切块/导入
│   ├── encryption/               # AES-GCM
│   ├── embedding/                # sentence-transformers embedding
│   ├── retrieval/                # Qdrant VectorStore + Retriever
│   ├── llm/                      # Ollama LLM 适配
│   ├── rag_pipeline/             # RAGSystem + rerank + prompts
│   ├── evaluation/               # metrics + benchmarking
│   └── utils.py                  # AuditLogger 等通用工具
├── scripts/                      # 实验/评估/诊断脚本
├── accuracy_test/                # 精度评估脚本（预测/打分/协议）
├── unencrypted/                  # 明文对照实验（性能对比）
├── config/config.yaml
├── data/
├── qdrant_storage/
├── logs/
└── streamlit_app.py
```

## 检索与 Rerank 说明（与代码一致）

### 检索结果 payload 约定

向量库中每个文本块 payload 主要包含：

- `ciphertext`：AES-GCM 密文（Base64 字符串）
- `nonce`：AES-GCM nonce（Base64 字符串）
- `metadata`：至少包含 `source_file`、`chunk_id`、`doc_id`，以及导入时计算的 `content_hash` 等

检索阶段对明文字段也做兼容（用于历史数据或对照实验）：`text/plaintext/content/source_text`。

读取优先级：
1. 若同时存在 `ciphertext` 和 `nonce`：解密得到明文
2. 否则若存在 `text/plaintext/content/source_text`：直接使用
3. 都没有：跳过该结果（不让整条链路失败）

### 本地 Rerank（两阶段检索）

- 默认实现：`LocalReranker`（纯本地、确定性、基于词重叠+时间/邮件头等先验，见 `src/rag_pipeline/rerank.py`）。
- 代码中也提供了 `HybridReranker`（优先尝试 `sentence-transformers` 的 CrossEncoder，失败则回退到 `LocalReranker`）。
- 主入口 `src/rag_pipeline/rag_system.py` 当前默认启用的是 **LocalReranker**（由 `config/config.yaml` 的 `rerank.enabled` 控制）。

## 配置说明

主配置文件：`config/config.yaml`（请以此为准）。

特别说明：
- `vector_db` 默认不配置 `host/port`，因此会走 **Qdrant 本地存储目录**模式（`storage_path: ./qdrant_storage`）。
- `llm.quantization` 字段仅用于实验记录/对齐；仓库不包含 bitsandbytes/QLoRA 推理路径，实际量化发生在 Ollama 侧（GGUF 模型变体）。

## 基准测试 / 评估

### 1) 基准测试（Benchmark）

脚本：`scripts/run_benchmark.py`

```powershell
python scripts\run_benchmark.py --test_queries data\test_datasets\test_queries.txt --benchmark_type k_values --output benchmark_results.json
python scripts\run_benchmark.py --test_queries data\test_datasets\test_queries.txt --benchmark_type embedding_models --output embedding_benchmark.json
python scripts\run_benchmark.py --test_queries data\test_datasets\test_queries.txt --benchmark_type full --output full_benchmark.json
```

### 2) 精度评估（accuracy_test）

- 生成预测：`accuracy_test/run_rag_predictions.py`
- 打分：`accuracy_test/score_rag_predictions.py`
- 端到端评估运行器：`accuracy_test/run_rag_accuracy_eval.py`

（具体协议与字段见 `accuracy_test/ACCURACY_EVAL_PROTOCOL.md` 与目录内 README。）

### 3) 明文对照性能实验（unencrypted）

- 明文导入：`unencrypted/ingest_plaintext.py`
- 明文 RAG：`unencrypted/build_plaintext_rag.py`
- 性能对比：`unencrypted/bench/run_perf_compare.py`、`unencrypted/bench/run_perf_compare_60q.py`
- 汇总：`unencrypted/results_60q/60q_test/summarize_perf_results.py`

## 安全性与隐私（按当前实现表述）

### 加密机制
- 算法：AES-256-GCM
- 存储：Qdrant payload 默认存密文与元数据；embedding 在本地对明文生成

### 审计日志（重要：与代码一致的口径）

当前实现会记录审计事件，并做链式哈希完整性校验。

- `AuditLogger.log_query()` 会写入：`query_length` 以及 `query` 前 **200** 字符作为预览（`src/utils.py`）。
- **不会**在审计事件里存储完整答案正文（当前主流程没有写入 answer 文本）。

> 如果你需要论文/合规层面的“完全不记录查询内容”，建议将 `log_query()` 改为仅记录哈希或仅记录长度；README 将以你代码实现为准同步更新。

## 故障排除

### Ollama 连接失败
```powershell
Invoke-WebRequest http://localhost:11434/api/tags -UseBasicParsing
ollama serve
```

### Qdrant 本地存储被占用（Windows 常见）
```powershell
# 退出占用 qdrant_storage 的其它进程/终端后重试。
# 或备份并重建目录（按需执行）：
Rename-Item qdrant_storage qdrant_storage_backup -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path qdrant_storage -Force | Out-Null
```

## 许可证

MIT License

## 致谢

- [Ollama](https://ollama.com/) - 本地LLM部署
- [Qdrant](https://qdrant.tech/) - 向量数据库
- [Sentence Transformers](https://www.sbert.net/) - Embedding模型
- [HuggingFace](https://huggingface.co/) - 模型与工具
