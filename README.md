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
│  │  Vector  │ <─── │ Embedding│ <─── │ Original │      │
│  │ Database │      │  Model   │      │   Text   │      │
│  │ (Qdrant) │      └──────────┘      └──────────┘      │
│  └──────────┘                                            │
│       │                                                   │
│       ▼                                                   │
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

## 技术栈

### 核心依赖
- **Python 3.8+**
- **Qdrant**: 向量数据库
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

```bash
# 克隆仓库
git clone https://github.com/Irene-hua/graduation-design.git
cd graduation-design

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 安装Ollama

```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# 启动Ollama服务
ollama serve

# 拉取模型（新终端）
ollama pull llama2
```

### 3. 文档导入

```bash
# 将文档放入 data/raw/ 目录
# 运行文档导入脚本
python scripts/ingest_documents.py \
  --input_dir data/raw/ \
  --config config/config.yaml \
  --generate_key
```

### 4. 运行RAG系统

```bash
# 交互式问答
python scripts/run_rag.py \
  --config config/config.yaml \
  --key_file encryption.key

# 单个问题
python scripts/run_rag.py \
  --question "什么是机器学习？" \
  --top_k 5
```

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
│   └── run_benchmark.py          # 运行基准测试
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
- 自动解密检索结果

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

在 `src/document_processing/document_parser.py` 中添加新的解析方法：

```python
def _parse_new_format(self, filepath: Path) -> str:
    # 实现新格式的解析逻辑
    pass
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

### Ollama连接失败
```bash
# 检查Ollama服务状态
ps aux | grep ollama

# 重启服务
ollama serve
```

### Qdrant存储错误
```bash
# 清空并重建集合
rm -rf qdrant_storage/
python scripts/ingest_documents.py --generate_key
```

### 内存不足
- 降低 `embedding.batch_size`
- 减小 `document_processing.chunk_size`
- 使用更小的Embedding模型
- 启用模型量化

### GPU相关问题
```bash
# 检查CUDA可用性
python -c "import torch; print(torch.cuda.is_available())"

# 强制使用CPU
# 在配置中设置 device: 'cpu'
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
