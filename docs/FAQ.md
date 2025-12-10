# 常见问题 (FAQ)

## 安装和设置

### Q1: 系统要求是什么？

**最低要求**:
- Python 3.8+
- 8GB RAM
- 10GB 可用磁盘空间
- CPU: 4核心

**推荐配置**:
- Python 3.10+
- 16GB+ RAM
- 50GB+ SSD
- NVIDIA GPU (可选，用于加速)
- 8核心+ CPU

### Q2: 安装失败怎么办？

尝试以下步骤：

```bash
# 1. 升级pip
pip install --upgrade pip

# 2. 单独安装可能有问题的包
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. 如果CUDA相关错误，安装CPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. 重新安装依赖
pip install -r requirements.txt
```

### Q3: Ollama安装在哪里？

- Linux/Mac: 参考 https://ollama.com/download
- Windows: 使用WSL2或Docker

## 使用问题

### Q4: 文档导入很慢怎么办？

优化方法：

```yaml
# 在config.yaml中调整
embedding:
  batch_size: 64  # 增大批次大小（如果内存足够）

document_processing:
  chunk_size: 256  # 减小块大小，生成更少的块
```

### Q5: 检索结果不相关怎么办？

1. **调整K值**：增加返回结果数量
```bash
python scripts/run_rag.py --top_k 10
```

2. **更换Embedding模型**：使用更大的模型
```yaml
embedding:
  model_name: 'sentence-transformers/all-MiniLM-L12-v2'
```

3. **优化切块**：减少chunk_size保持上下文完整性

### Q6: LLM生成的答案质量不高？

1. **调整temperature**：
```bash
python scripts/run_rag.py --temperature 0.3  # 更确定性
python scripts/run_rag.py --temperature 0.9  # 更有创意
```

2. **更换模型**：
```bash
ollama pull mistral  # 更强大的模型
```

3. **优化提示词**：修改config.yaml中的prompt_template

### Q7: 如何支持中文文档？

系统已支持中文，但建议：

1. 使用多语言Embedding模型：
```yaml
embedding:
  model_name: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
```

2. 使用支持中文的LLM：
```bash
ollama pull qwen  # 千问模型
```

## 性能问题

### Q8: 内存占用太大？

**减少内存占用**：

1. 启用模型量化：
```yaml
llm:
  quantization:
    enabled: true
    bits: 4
```

2. 减小batch_size：
```yaml
embedding:
  batch_size: 16  # 或更小
```

3. 使用更小的模型：
```bash
ollama pull phi  # 很小的模型
```

### Q9: 响应速度慢？

**加速方法**：

1. 使用GPU（如有）：
```python
model = EmbeddingModel('model-name', device='cuda')
```

2. 减少top_k：
```bash
python scripts/run_rag.py --top_k 3
```

3. 启用缓存：
```yaml
performance:
  enable_caching: true
```

### Q10: GPU内存不足？

使用CPU模式：
```bash
export CUDA_VISIBLE_DEVICES=""  # 禁用GPU
python scripts/run_rag.py
```

或者使用量化：
```yaml
llm:
  quantization:
    enabled: true
    bits: 4  # 极大减少显存占用
```

## 数据和安全

### Q11: 加密密钥丢失了怎么办？

**无法恢复**。加密密钥丢失意味着：
- 无法解密已存储的数据
- 需要重新导入所有文档

预防措施：
```bash
# 备份密钥
cp encryption.key encryption.key.backup
```

### Q12: 如何更换加密密钥？

```bash
# 1. 删除旧数据
rm -rf qdrant_storage/

# 2. 生成新密钥并重新导入
python scripts/ingest_documents.py \
  --input_dir data/raw/ \
  --generate_key
```

### Q13: 数据存储在哪里？

```
qdrant_storage/     # 向量数据库（含加密数据）
encryption.key      # 加密密钥
logs/              # 审计日志
```

## 错误处理

### Q14: "Collection not found" 错误？

运行文档导入：
```bash
python scripts/ingest_documents.py --input_dir data/raw/
```

### Q15: "Model not found" 错误？

拉取模型：
```bash
ollama pull llama2
```

### Q16: "Connection refused" 错误？

启动Ollama服务：
```bash
ollama serve
```

### Q17: PDF解析失败？

确保安装了pypdf：
```bash
pip install pypdf
```

对于加密的PDF，需要先解密。

## 高级用法

### Q18: 如何集成到自己的应用？

```python
from src.rag_pipeline import RAGSystem
# ... 初始化组件
rag = RAGSystem(retriever, llm_client)
result = rag.answer_question("Your question")
```

详见 [API文档](API.md)

### Q19: 如何批量处理问题？

```python
questions = ["Q1", "Q2", "Q3"]
results = rag_system.batch_answer(questions)
```

### Q20: 如何自定义提示词？

修改 config.yaml：
```yaml
rag:
  prompt_template: |
    你的自定义模板
    上下文: {context}
    问题: {question}
    回答:
```

### Q21: 如何评估系统性能？

```bash
# 运行基准测试
python scripts/run_benchmark.py \
  --test_queries data/test_datasets/test_queries.txt \
  --benchmark_type full
```

### Q22: 如何监控系统？

查看审计日志：
```bash
tail -f logs/audit_*.log
```

获取统计信息：
```python
from src.audit import AuditLogger
logger = AuditLogger()
stats = logger.get_statistics()
print(stats)
```

## 贡献和反馈

### Q23: 如何报告bug？

在GitHub上创建Issue：
https://github.com/Irene-hua/graduation-design/issues

### Q24: 如何贡献代码？

1. Fork仓库
2. 创建feature分支
3. 提交Pull Request

### Q25: 在哪里获取更多帮助？

- 📚 [README](../README.md)
- 🏗️ [架构文档](ARCHITECTURE.md)
- 🔧 [API文档](API.md)
- 🚀 [快速开始](../QUICKSTART.md)

---

问题未解决？[提交Issue](https://github.com/Irene-hua/graduation-design/issues)
