# 第4章 系统实现

## 4.1 开发环境配置

### 4.1.1 软件环境

本系统在Windows 11环境下开发，Python版本为3.10.13。系统基于Python的模块化设计，各核心模块通过明确的接口相互调用。开发工具包括PyCharm IDE、Git版本控制。

### 4.1.2 技术栈与依赖

**核心依赖库**：
- **qdrant-client 0.11+**：向量数据库的Python客户端
- **sentence-transformers 2.2+**：Embedding模型框架和预训练模型
- **cryptography 41.0+**：AES-256-GCM加密的Python原语实现
- **PyYAML 6.0+**：配置文件解析
- **pypdf 3.0+**：PDF文档解析
- **python-docx 0.8+**：DOCX文档解析
- **requests 2.28+**：HTTP客户端（与Ollama通信）

**部署环境**：
- **Qdrant**：版本1.7.0以上，本地模式运行，默认监听6333端口提供HTTP API
- **Ollama**：版本0.1.20以上，本地LLM部署框架，默认监听11434端口

### 4.1.3 LLM部署与量化

本系统采用Ollama作为本地大语言模型部署框架。安装完成后，通过`ollama serve`命令启动服务，Ollama通过localhost:11434提供API服务。

系统选用Mistral-7B-Instruct模型的4-bit NF4量化版本作为默认LLM。量化过程将原始模型（约14GB FP16）压缩至约4GB，压缩率达71%。量化后的模型能够在16GB内存的消费级设备上运行，无需独立显卡。

---

## 4.2 核心模块实现

### 4.2.1 加密模块实现

**实现文件**：`src/encryption/aes_encryption.py`

**实现目标**：将第3章3.3.1-3.3.2节设计的混合加密策略和密钥管理机制转化为可执行代码。

**关键实现决策**：

(1) **密钥派生策略**：采用PBKDF2算法从用户口令派生加密密钥。参数设置为：
- 哈希算法：SHA256
- 密钥长度：256位（32字节）
- 迭代次数：100000
- 盐值：随机生成（16字节）

为避免早期实现中硬编码盐值的安全隐患，采用随机盐值并在首次派生时持久化存储，符合密码学最佳实践。

(2) **加密数据格式**：严格遵循第3章设计，每个加密块包含密文、随机Nonce和认证标签三个字段，确保数据完整性和可验证性。系统使用Python `cryptography`库的`AESGCM`类实现，该类自动处理认证标签的生成和验证。

(3) **异常处理机制**：解密操作包含完整的异常处理，能够识别密钥错误、数据篡改等情况，并给出明确的错误提示。

**核心类设计**：

```python
class AESEncryption:
    def __init__(self, key_size: int = 256):
        """初始化加密器，默认256位密钥"""
    
    def encrypt(self, plaintext: str) -> Tuple[str, str]:
        """返回 (密文_Base64, nonce_Base64)"""
        # 1. 生成随机Nonce（12字节）
        # 2. 使用AES-GCM加密明文
        # 3. 返回Base64编码的密文和Nonce
    
    def decrypt(self, ciphertext_b64: str, nonce_b64: str) -> str:
        """解密Base64编码的密文，返回明文"""
        # 1. Base64解码密文和Nonce
        # 2. 使用AES-GCM解密（自动验证认证标签）
        # 3. 解码UTF-8得到明文
    
    def derive_key_from_password(self, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """使用PBKDF2从口令派生密钥"""
        # 参数：100000迭代，SHA256，随机盐值
    
    def load_key(self, filepath: str):
        """从文件加载密钥"""
    
    def save_key(self, filepath: str):
        """将密钥保存至文件"""
```

### 4.2.2 检索模块实现

**实现文件**：`src/retrieval/retriever.py`

**实现目标**：实现第3章3.5.1节设计的检索功能，对向量检索的候选结果进行解密和优化。

**关键实现决策**：

(1) **两阶段密钥恢复**：检索结果首先尝试从Payload的明文字段（`text`、`plaintext`、`content`等）读取。若不存在这些字段，则尝试解密加密字段（`ciphertext`+`nonce`）。这种设计支持加密和非加密Payload共存，提升系统灵活性，也为混合测试（加密vs明文）提供了基础。

(2) **针对性的后备检索方案**：对于时间相关和邮件查询，系统实现了基于电子邮件头部特征的优化。实现包括：
- `_looks_like_time_question()`：识别时间相关查询（包含"date"、"time"、"when"、"responded"等关键词）
- `_looks_like_email_thread()`：识别邮件相关查询（包含"email"、"subject"等关键词）
- `_header_boost_fallback()`：对于时间/邮件查询，若向量检索结果中不包含邮件头部信息，则进行有限的本地扫描（上限约800-1000条记录）以发现包含时间戳的块

(3) **增强的查询理解**：系统通过查询内容识别其类型，进而采取对应的检索策略，提升特定类型查询的准确率。

**核心方法**：

```python
class Retriever:
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        执行完整的检索流程：
        1. 将查询转换为Embedding向量
        2. 在向量库中进行Top-K检索
        3. 解密候选结果的Payload
        4. 对于时间/邮件查询，执行可选的后备检索
        
        返回：List[Dict]，每个字典包含：
        - text：解密后的文本
        - score：相似度分数
        - metadata：元数据（source_file、chunk_id等）
        """
    
    def _extract_plaintext_from_payload(self, payload: Dict) -> Optional[str]:
        """从Payload中提取明文（支持明文字段或解密密文）"""
    
    def _looks_like_time_question(self, query: str) -> bool:
        """识别时间相关查询"""
    
    def _header_boost_fallback(self, query: str, top_k: int) -> List[Dict]:
        """对时间/邮件查询执行优化的后备检索"""
```

### 4.2.3 RAG主流程实现

**实现文件**：`src/rag_pipeline/rag_system.py`

**实现目标**：整合各核心模块，实现第3章3.5节设计的在线问答完整流程和两阶段拒答机制。

**关键实现决策**：

(1) **灵活的Prompt管理**：系统支持可配置的Prompt模板。默认Prompt包含以下关键指示：
- 进行推理和上下文连接
- 处理特定类型的问题（如WHO问题需要列举具体名词而不是模糊表述）
- 明确限制：仅基于提供的文档回答
- 拒答引导：当信息不足时回复"I don't know"

(2) **两阶段拒答实现**：
- **第一阶段**：检索后若无有效上下文（所有候选块均无法解密或被过滤），直接拒答，不调用LLM
- **第二阶段**：LLM生成后，通过`is_valid_answer()`方法检测低质量答案（空、过短<20字符、包含"I don't know"等）

(3) **WHO问题的特殊处理**：系统检测WHO问题，若LLM首次回答不具体（包含"all characters"、"everyone"等模糊表述），则进行一次重试，在Prompt中增加"列出具体名词"的明确指示。这个重试机制仅针对WHO问题，不影响其他问题的处理。

(4) **置信度计算**：实现第3章设计的置信度公式：

$$C = 0.6 \times S_{\text{rerank}} + 0.2 \times L + 0.2 \times Q$$

其中$S_{\text{rerank}}$为重排序得分，$L$为答案长度因子，$Q$为输出质量因子。置信度用于实验评估，不用于实际拒答决策。

(5) **答案规范化处理**：在返回答案前，系统通过`_normalize_to_answer_format()`方法将答案格式统一为"Answer: <内容>"，移除内部RAG标记（如"Context: [4]"），确保输出的一致性。

(6) **异常处理覆盖**：主流程包含完整的异常处理，覆盖密钥错误、解密失败、向量库异常、LLM服务不可用等失败情况，确保系统健壮性。

**核心类设计**：

```python
class RAGSystem:
    def answer_question(self, question: str, top_k: int = 5, 
                       temperature: float = 0.2) -> Dict:
        """
        执行完整的问答流程，返回字典包含：
        - answer：最终答案（规范化格式）
        - confidence：置信度分数（用于评估）
        - path：推理路径（"FAIL"或"RAG"）
        - reasoning_path：详细推理路径
        - retrieval_time：检索耗时
        - generation_time：生成耗时
        - used_chunks：用于生成答案的文本块
        - weak_answer：是否为低质量答案
        - retrieval_empty：是否无检索结果
        """
    
    @staticmethod
    def is_valid_answer(answer: str) -> bool:
        """判断答案是否有效（非低质量答案）"""
        # 规则：非空、不完全是"I don't know"等
    
    @staticmethod
    def _compute_confidence(...) -> float:
        """计算置信度分数（用于评估）"""
        # 公式：0.6*S_rerank + 0.2*L + 0.2*Q
    
    @staticmethod
    def _looks_like_who_question(question: str) -> bool:
        """检测WHO问题"""
    
    @staticmethod
    def _has_vague_who_answer(answer: str) -> bool:
        """检测WHO问题的模糊回答"""
```

### 4.2.4 文档处理与向量存储模块

**实现文件**：`src/document_processing/`、`src/retrieval/vector_store.py`

**文档处理模块**：
- **`DocumentParser`**：支持TXT、PDF、DOCX、Markdown四种格式的解析，采用策略模式根据文件类型动态选择解析器
- **`TextChunker`**：实现递归字符分块，参数为chunk_size=512、chunk_overlap=50

**向量存储模块**：
- **`VectorStore`**：Qdrant客户端的封装，提供统一接口
  - 支持集合创建、批量插入、Top-K检索
  - 向量和Payload分离存储
  - 支持加密和明文Payload共存

### 4.2.5 审计日志模块实现

**实现文件**：`src/audit/`

**核心特性**：

(1) **链式哈希完整性保护**：严格实现第3章设计。每条日志记录包含当前事件的SHA256哈希值以及前一条日志的哈希值，形成链式结构。提供`verify_integrity()`方法供手动验证日志完整性。

(2) **隐私保护**：审计日志仅记录操作元信息，不记录完整内容。查询内容在日志中截断至前200字符，既保留基本特征用于审计，又避免敏感信息泄露。

(3) **存储隔离**：审计日志文件存储在独立的`./logs/`目录下，与向量数据库文件、密钥文件在存储位置上相互隔离，实现安全域分离。

(4) **可配置性**：通过配置文件控制开启/关闭，支持设置日志级别、完整性检查、日志轮转策略等参数。

**核心类设计**：

```python
class AuditLogger:
    def log_system_access(self, user: str, action: str):
        """记录系统访问事件"""
    
    def log_query(self, query: str, truncate_length: int = 200):
        """记录查询操作（截断敏感内容）"""
    
    def log_model_invocation(self, model_name: str, inference_time: float):
        """记录模型调用"""
    
    def log_error(self, context: str, error_message: str):
        """记录错误信息"""
    
    def verify_integrity(self) -> bool:
        """验证日志链式哈希完整性"""
```

---

## 4.3 系统集成与测试

### 4.3.1 知识库构建脚本

**脚本**：`src/document_processing/ingest.py`

**功能**：将第3章3.4节设计的知识库构建流程封装为可执行脚本。

**关键实现**：
- 批量文档导入：扫描指定目录下的所有支持格式文档
- 逐个处理：解析、分块、加密、向量化
- 批量写入：通过upsert操作批量写入向量数据库
- 错误隔离：单文档失败不影响其他文档处理
- 进度反馈：输出处理进度（已处理文档数、生成块数）

**使用示例**：

```bash
python -m src.document_processing.ingest \
  --input_dir data/raw \
  --collection_name encrypted_documents \
  --key_file encryption.key \
  --config config/config.yaml
```

### 4.3.2 批量评测脚本

**脚本**：`accuracy_test/run_rag_accuracy_eval.py`

**功能**：为第5章实验提供系统化的评测支持。

**关键实现**：
- 加载测试数据集（含Gold Answer）
- 逐个问题调用RAG系统获取回答
- 记录响应时间、成功标志等元数据
- 调用指标计算模块计算F1、Precision、Recall等
- 结果持久化（JSON/CSV格式）

### 4.3.3 可视化界面

**技术**：基于Streamlit框架

**功能**：
- 交互式问答：用户输入问题，系统实时返回答案
- 参数配置：侧边栏支持调整系统参数
- 结果可视化：展示检索块、相似度分数等信息

---

## 4.4 系统测试

### 4.4.1 单元测试

针对核心模块编写单元测试，覆盖主要功能路径和典型边界条件。

**测试覆盖情况**：

| 测试模块 | 核心测试内容 | 关键边界条件 |
|---------|-------------|------------|
| 加密模块 | 加密/解密对称性、密钥派生、篡改检测 | 空字符串、错误密钥、数据篡改 |
| 检索模块 | 集合创建、向量插入、Top-K检索 | 空集合、异常连接、解密失败 |
| 文档处理 | 多格式解析、分块功能 | 空文件、不支持格式、损坏文件 |
| RAG主流程 | 查询流程、拒答机制、WHO处理 | 空知识库、低置信度、WHO问题 |

**加密模块测试示例**：
- 加密解密对称性验证
- 不同长度文本的加密测试
- 空字符串处理
- 错误密钥解密失败验证
- 数据篡改检测（AESGCM认证标签验证）

### 4.4.2 集成测试

设计端到端集成测试，验证多模块协同工作的正确性。

**测试场景**：

(1) **端到端查询测试**：构建测试知识库，执行完整查询流程，验证从文档导入到答案生成的全链路正确性。

(2) **空知识库测试**：在未构建知识库的情况下执行查询，验证系统能够正确触发拒答并给出明确提示。

(3) **密钥错误测试**：使用错误密钥初始化系统，验证解密失败时系统能够正确处理异常。

(4) **WHO问题处理测试**：验证WHO问题的特殊处理和重试机制是否有效工作。

集成测试采用临时目录隔离测试数据，每个测试用例独立设置和清理，确保测试之间互不干扰。

---

## 4.5 本章小结

本章完成了第3章设计方案的代码实现，将设计决策转化为具体的可执行系统。

**实现成果包括**：

(1) **加密模块**：实现了PBKDF2密钥派生（100000迭代、随机盐值）和AES-256-GCM加密，修复了早期硬编码盐值的安全隐患

(2) **检索模块**：实现了两阶段密钥恢复、查询优化（时间/邮件查询）和有限的本地后备检索

(3) **RAG主流程**：整合各子模块，实现了完整的在线问答流程、两阶段拒答机制和WHO问题的特殊处理

(4) **文档处理**：实现了多格式解析和递归字符分块

(5) **审计日志**：实现了链式哈希完整性保护和隐私保护设计

(6) **系统集成**：提供了知识库构建脚本、批量评测脚本和Web UI界面

(7) **测试方案**：覆盖了单元测试和集成测试

上述实现严格遵循第3章的设计方案，为第5章的系统验证提供了可靠的系统基础。
