使用说明：src.document_processing.ingest

简短说明

该模块将文档解析、切分、加密、嵌入并写入向量数据库的整个导入流程封装为库函数，方便在代码中以可编程方式调用。核心函数为 `ingest_documents`，也提供 `main()` 用于命令行调用（如果你想保留 CLI，可自行用小脚本调用）。

快速示例（Python 调用）

- 在 Python 中直接调用库接口（推荐）：

```python
from src.document_processing import ingest_documents

result = ingest_documents(
    input_dir='data/raw',
    config_path='config/config.yaml',
    key_file='encryption.key',
    generate_key=False,
    collection_name=None,
    reset_collection=False,
    log_file=None,
)
print(result)
```

- 通过一行命令从 shell/PowerShell 调用（不依赖脚本文件）：

PowerShell 或 CMD:

```powershell
python -c "from src.document_processing import ingest_documents; ingest_documents('data/raw','config/config.yaml','encryption.key')"
```

类 Unix shell 或 Windows PowerShell（直接以模块运行 main 若可用）：

```bash
python -m src.document_processing.ingest --input_dir data/raw --config config/config.yaml --key_file encryption.key
```

参数说明（常用）

- `--input_dir` : 要导入的文档目录（必需）。
- `--config` : 配置文件路径，默认 `config/config.yaml`，包含 chunk 大小、嵌入模型、向量库路径等设置。
- `--key_file` : 加密密钥文件，默认 `encryption.key`。若使用 `--generate_key` 会生成并保存新密钥。
- `--generate_key` : 生成新的加密密钥并写入 `key_file`。
- `--collection_name` : 覆盖配置中的向量集合名。
- `--reset_collection` : 在导入前重建目标集合（会删除原集合中所有数据）。
- `--log_file` : 可选，追加写入导入记录（YAML/JSONL）。

注意事项

- `ingest_documents` 会依赖 `src.document_processing.DocumentParser`, `TextChunker`, `AESEncryption`, `EmbeddingModel`, `VectorStore` 等模块，请确保环境中已安装相应第三方依赖（例如 `requests`, `pypdf`, `python-docx`, `qdrant-client` 等，视文件与向量库后端而定）。
- 该函数会触发实际的写入到向量库（可能是本地 qdrant 存储），在测试或 dry-run 时请使用隔离目录或 mock。
