# 加密机制对RAG系统性能影响的对比实验（可复现实验文档）

> 本文档描述一个**对比/消融实验**：在保持 RAG 管线其它部分完全一致的前提下，仅移除“文本加密存储”机制，构造一个未加密（明文payload）本地轻量 RAG 基线系统，并对两者在**响应速度**与**资源占用**上的差异进行评估。
>
> 本实验的实现保证：**不修改、不影响**现有加密RAG系统；未加密基线的所有新增代码与产物均放置在 `unencrypted/` 目录下，且使用独立的本地 Qdrant 存储目录与 collection。

---

## 1. 实验目的与研究问题

### 1.1 目的
评估“文档加密存储（AES-GCM）”对本地部署 RAG 系统以下方面的影响：
- **响应速度**：检索阶段、生成阶段、端到端总耗时。
- **资源占用**：CPU 使用率、内存（RSS）峰值/均值、向量库（Qdrant 本地存储）磁盘占用作为索引大小近似。

### 1.2 研究问题（RQ）
- **RQ1**：在相同的 embedding/检索/重排/LLM 设置下，密文payload导致的**额外解密成本**会让检索耗时上升多少？
- **RQ2**：加密是否会影响生成阶段耗时？（预期影响较小，主要影响检索阶段）
- **RQ3**：密文payload相对明文payload是否显著增加 Qdrant 的存储体积（数据库/索引占用）？

---

## 2. 实验变量与控制

### 2.1 自变量（唯一变化点）
- **Encrypted（加密RAG）**：写入 Qdrant 的 payload 为 `ciphertext` + `nonce`（密文与随机数），检索后在 `Retriever` 中解密得到 `text`。
- **Plaintext（未加密RAG）**：写入 Qdrant 的 payload 直接存储 `text`（明文），检索后无需解密。

### 2.2 控制变量（保证公平对比）
两套系统保持完全一致：
- 文档解析与切分：`DocumentParser`、`TextChunker`（chunk_size / overlap 来自同一 `config/config.yaml`）
- embedding：同一 `EmbeddingModel` 与 batch_size
- vector schema：同样的向量维度与距离度量
- 检索逻辑：同一 `Retriever.retrieve()` 返回格式（均输出 `text` + `metadata`）
- reranker 与 prompt：同一配置
- LLM：同一 Ollama 模型与 base_url
- 查询集合：同一 `unencrypted/bench/queries_lihua_world.json`

---

## 3. 数据集与导入

### 3.1 原始数据
使用目录：`data/raw/LiHua-World/` 作为导入原始文本。

### 3.2 加密系统导入（已存在，不改动）
使用项目自带脚本（如已导入可跳过）：
- `scripts/ingest_documents.py` 会对 chunk 文本执行 AES-GCM 加密后写入配置中的加密 collection。

### 3.3 未加密基线导入（新增，隔离）
新增脚本：`unencrypted/ingest_plaintext.py`
- 写入独立存储目录：`unencrypted/qdrant_storage_plaintext/`
- collection 名：默认 `plaintext_documents_lihua_world`

---

## 4. 性能指标定义

### 4.1 延迟类指标（Latency）
统一以 `RAGSystem.answer_question()` 输出为准：
- `retrieval_time`：从开始到检索/（可选重排）完成的耗时
- `generation_time`：LLM 生成耗时
- `total_time`：端到端耗时

统计方法：对每个指标报告 mean / median / p90 / p95 / min / max。

### 4.2 资源占用（Resource usage）
- CPU：查询过程中脚本进程采样的 `cpu_percent_avg` / `cpu_percent_max`（需要 `psutil`）
- 内存：脚本进程 RSS 的 `rss_bytes_avg` / `rss_bytes_max`
- 存储占用：Qdrant storage 路径目录总大小（近似数据库/索引占用）

> 注：存储目录大小是“可复制、面向论文”的近似指标，能稳定反映 payload 增大与索引文件增量。

---

## 5. 实验流程（全过程，可复现）

### 5.1 环境准备
- Python 版本：与项目一致
- 依赖：`requirements.txt`
- 本地服务：Ollama（用于生成）

为了采集 CPU/内存建议安装：`psutil`（若未安装脚本也可运行，但资源指标为 n/a）。

### 5.2 导入阶段
1) **加密系统导入**（如已导入可跳过）
- 输入目录：`data/raw/LiHua-World`
- 结果写入：配置文件中的 `vector_db.storage_path` 与 `vector_db.collection_name`

2) **未加密基线导入**（必须）
- 脚本：`unencrypted/ingest_plaintext.py`
- 输入目录：`data/raw/LiHua-World`
- 输出：`unencrypted/qdrant_storage_plaintext` + `plaintext_documents_lihua_world`

### 5.3 评测阶段（对比实验）
- 脚本：`unencrypted/bench/run_perf_compare.py`
- queries：`unencrypted/bench/queries_lihua_world.json`
- 每个 query 各运行一次（加密/明文各跑一遍）

输出目录：`unencrypted/results/<run_id>/`
- `samples.csv`：逐query原始指标
- `samples.jsonl`
- `summary.json`：统计汇总
- `REPORT.md`：汇总对比报告（可直接引用到论文）

---

## 6. 风险与规避

- **不影响现有向量库**：明文基线使用独立 storage 路径与 collection，且所有新增文件位于 `unencrypted/`。
- **公平性**：未加密只改变 payload 字段，不改变 chunking/embedding/retrieval/rerank/LLM。
- **噪声控制**：建议固定机器状态、关闭后台大型任务、设置 `--sleep_s` 以降低热噪声；可多次运行取均值。
- **可重复性**：记录 config、collection、storage_path、n_queries 等到 summary.json。

---

## 7. 结果解读模板（写论文可直接用）

运行后打开 `unencrypted/results/<run_id>/REPORT.md`。

建议在论文中按如下逻辑写“结论”：
1) 观察 `retrieval_time_s` 的 median/p95 差异，说明加密主要影响检索阶段（payload解密与字段处理）。
2) 观察 `generation_time_s` 的差异通常较小，证明加密机制对 LLM 生成阶段影响不显著。
3) 对比 Qdrant 存储目录大小，讨论密文payload带来的存储膨胀。

---

## 8. 文件与实现位置（对应仓库）

- 加密导入：`scripts/ingest_documents.py`
- 加密检索与解密：`src/retrieval/retriever.py`
- 明文导入（新增）：`unencrypted/ingest_plaintext.py`
- 明文运行时构建（新增）：`unencrypted/build_plaintext_rag.py`
- 对比基准（新增）：`unencrypted/bench/run_perf_compare.py`


