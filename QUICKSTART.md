# 快速开始指南

## 5分钟上手

### 第一步：环境准备

```bash
# 克隆仓库
git clone https://github.com/Irene-hua/graduation-design.git
cd graduation-design

# 运行自动化设置脚本
bash scripts/setup.sh
```

### 第二步：安装Ollama

```bash
# Linux/Mac安装Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 启动Ollama服务（保持运行）
ollama serve &

# 拉取模型（新终端）
ollama pull llama2
```

### 第三步：准备文档

```bash
# 将你的文档放入 data/raw/ 目录
# 支持的格式: .txt, .pdf, .docx, .md
cp your_documents.txt data/raw/
```

或者使用提供的示例文档：
```bash
# 已包含示例文档
ls data/raw/sample_document.txt
```

### 第四步：导入文档

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行文档导入脚本
python scripts/ingest_documents.py \
  --input_dir data/raw/ \
  --generate_key

# 这将：
# 1. 解析所有文档
# 2. 切分为小块
# 3. 加密每个块
# 4. 生成向量表示
# 5. 存储到本地向量数据库
```

### 第五步：开始使用

```bash
# 交互式问答模式
python scripts/run_rag.py

# 或者直接问单个问题
python scripts/run_rag.py --question "什么是机器学习？"
```

## 示例会话

```
$ python scripts/run_rag.py

==================================================
Interactive RAG Q&A System
Type 'quit' or 'exit' to stop
==================================================

Your question: 什么是机器学习？
--------------------------------------------------

Answer: 机器学习是人工智能的一个子集，专注于开发能够从数据中学习的算法。
它使计算机能够在没有明确编程的情况下改进性能。

Retrieved 5 chunks
Retrieval time: 0.234s
Generation time: 1.567s
Total time: 1.801s

Sources:
  1. sample_document.txt (score: 0.856)
  2. sample_document.txt (score: 0.792)
  3. sample_document.txt (score: 0.743)

Your question: quit
Goodbye!
```

## 常见问题

### Q: Ollama连接失败怎么办？

A: 确保Ollama服务正在运行：
```bash
# 检查服务
curl http://localhost:11434/api/tags

# 如果失败，重启服务
ollama serve
```

### Q: 内存不足怎么办？

A: 可以：
1. 使用更小的模型：`ollama pull phi`
2. 减小chunk_size（在config.yaml中）
3. 减小embedding batch_size

### Q: 如何更换LLM模型？

A: 修改 `config/config.yaml`：
```yaml
llm:
  model_name: 'mistral'  # 或 'phi', 'llama2'等
```

然后拉取新模型：
```bash
ollama pull mistral
```

### Q: 如何添加更多文档？

A: 只需：
```bash
# 1. 添加新文档到data/raw/
cp new_docs/*.pdf data/raw/

# 2. 重新运行导入（使用现有密钥）
python scripts/ingest_documents.py \
  --input_dir data/raw/ \
  --key_file encryption.key
```

## 下一步

- 📚 阅读完整[README](README.md)了解所有功能
- 🏗️ 查看[架构文档](docs/ARCHITECTURE.md)理解系统设计
- 🔧 查看[API文档](docs/API.md)学习编程接口
- 📊 运行基准测试评估性能：
  ```bash
  python scripts/run_benchmark.py \
    --test_queries data/test_datasets/test_queries.txt \
    --benchmark_type k_values
  ```

## 需要帮助？

- 检查系统设置：`python scripts/validate_setup.py`
- 查看日志：`tail -f logs/audit_*.log`
- 提交Issue：[GitHub Issues](https://github.com/Irene-hua/graduation-design/issues)

祝使用愉快！🚀
