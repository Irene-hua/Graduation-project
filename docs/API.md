# API文档

## 核心类API

### AESEncryption

```python
from src.encryption import AESEncryption

# 初始化
encryption = AESEncryption(key_size=256)

# 生成密钥
encryption.generate_key()

# 加密
ciphertext, nonce = encryption.encrypt("text to encrypt")

# 解密
plaintext = encryption.decrypt(ciphertext, nonce)

# 保存/加载密钥
encryption.save_key("key.bin")
encryption.load_key("key.bin")
```

### EmbeddingModel

```python
from src.embedding import EmbeddingModel

# 初始化
model = EmbeddingModel('sentence-transformers/all-MiniLM-L6-v2')

# 编码文本
embeddings = model.encode(["text1", "text2"])

# 计算相似度
similarity = model.cosine_similarity(vec1, vec2)
```

### VectorStore

```python
from src.retrieval import VectorStore

# 初始化
store = VectorStore(
    collection_name='docs',
    dimension=384,
    storage_path='./qdrant'
)

# 添加向量
store.add_vectors(vectors, encrypted_chunks, metadata)

# 搜索
results = store.search(query_vector, top_k=5)
```

### RAGSystem

```python
from src.rag_pipeline import RAGSystem

# 初始化（需要retriever和llm_client）
rag = RAGSystem(retriever, llm_client)

# 回答问题
result = rag.answer_question(
    question="What is machine learning?",
    top_k=5
)

print(result['answer'])
print(result['total_time'])
```

## 命令行工具

### 文档导入

```bash
python scripts/ingest_documents.py \
  --input_dir data/raw/ \
  --config config/config.yaml \
  --generate_key
```

### 运行RAG

```bash
# 交互式
python scripts/run_rag.py

# 单个问题
python scripts/run_rag.py --question "Your question"
```

### 基准测试

```bash
python scripts/run_benchmark.py \
  --test_queries test.txt \
  --benchmark_type k_values
```

更多详细API文档请参考源代码注释。
