# 加密RAG vs 明文RAG：三类测试集（Multi/Single/Null）性能对比实验文档（可直接用于论文）

> 目标：在**不影响、且不破坏现有加密RAG系统**的前提下，构建一个“未加密的本地轻量RAG系统”（除 payload 明文存储外，其余模块与本项目保持一致），并在 3 个标准测试集（Multi/Single/Null，各 60 题）上对比**响应速度**与**资源占用**，给出可复现的实验流程、数据产物与结论报告。

---

## 1. 实验对照组设计

### 1.1 控制变量原则

为保证公平对比，除“是否对文本 payload 加密存储”之外，其余要素尽量保持一致：

- 相同的文本语料来源：`data/raw/LiHua-World`
- 相同的文本解析与切分策略：沿用 `config/config.yaml` 的 chunk_size 与 chunk_overlap
- 相同的 embedding 模型：`sentence-transformers/all-MiniLM-L6-v2`
- 相同的向量库（Qdrant）与相同 distance metric：Cosine
- 相同的检索参数（top_k=5）与相同 prompt 约束（Answer-only，禁止猜测）
- 相同的运行环境（同一台机器、同一时间段、尽量关闭后台干扰任务）

### 1.2 实验组与对照组

**A. 加密RAG（实验组 / Encrypted）**
- 使用项目现有 RAG 系统（`src/rag_pipeline/rag_system.py`）
- Qdrant payload 存储密文（ciphertext + nonce + 元数据）
- 查询时对返回 chunk 进行解密后组装上下文

**B. 明文RAG（对照组 / Plaintext）**
- 使用 `unencrypted/` 下的“明文版 runtime”（与现有 pipeline 结构一致）
- Qdrant payload 直接存储原始 chunk 文本字段 `text`
- 使用**独立的本地 Qdrant storage**：`unencrypted/qdrant_storage_plaintext`
- 使用独立 collection：`plaintext_documents_lihua_world`

> 关键要求：明文实验不会修改 `config/config.yaml`，不会覆盖原来的 `qdrant_storage/`，不会改变现有加密 collection。

---

## 2. 数据集与任务定义

### 2.1 语料（Index Corpus）
- 原始文本目录：`data/raw/LiHua-World`

### 2.2 测试集（Query Sets）
三类测试集各 60 题：

- **Multi**：`data/test_datasets/lihua-queries1`
  - Gold：`data/gold-answer/lihua-queries1-gold-answer`
- **Single**：`data/test_datasets/lihua-queries2`
  - Gold：`data/gold-answer/lihua-queries2-gold-answer`
- **Null**：`data/test_datasets/lihua-queries3`
  - Gold：`data/gold-answer/lihua-queries3-gold-answer`

本次性能评估只评估 **latency + resource usage**。正确性（PRF1 等）可继续沿用 `accuracy_test/` 的脚本体系。

---

## 3. 实验指标（Metrics）

### 3.1 延迟指标（Latency）
针对每个问题记录：

- `retrieval_time_s`：向量检索 + 候选 chunk 拉取（含 payload 解密/拼接的影响会间接反映）
- `generation_time_s`：LLM 生成耗时
- `total_time_s`：端到端总耗时（如底层返回）

统计维度：
- 每个测试集分别统计 mean/median/p90/p95
- 加密 vs 明文按 median 计算比值：`ratio = encrypted_median / plaintext_median`

### 3.2 资源指标（Resources）
- 峰值 RSS（需要 `psutil`）：`rss_bytes_max`
- CPU percent（粗粒度，psutil 可用时）：`cpu_percent_max`
- Qdrant 存储目录大小：`storage_size_bytes`

> 说明：本指标是**工程层面的可复现实验指标**，用于论文中讨论加密机制带来的开销与收益权衡。

---

## 4. 实验实现（本仓库已落地的脚本与产物）

### 4.1 明文索引构建（仅写入 unencrypted/）
脚本：`unencrypted/ingest_plaintext.py`

功能：
- 解析 `data/raw/LiHua-World`
- chunk 化
- embedding
- 写入本地 Qdrant：`unencrypted/qdrant_storage_plaintext`
- collection 默认：`plaintext_documents_lihua_world`

（PowerShell 示例）
```powershell
$env:PYTHONUTF8=1
python D:\PycharmProjects\Graduation-project\unencrypted\ingest_plaintext.py --input_dir D:\PycharmProjects\Graduation-project\data\raw\LiHua-World --reset_collection
```

预期输出关键字：
- `PLAINTEXT ingest done. points=...`
- `collection_info={'name': 'plaintext_documents_lihua_world', 'points_count': ...}`

### 4.2 60题三测试集性能对比
脚本：`unencrypted/bench/run_perf_compare_60q.py`

功能：
- 加载三套 60Q 测试集（Multi/Single/Null）
- 运行 **加密RAG** 与 **明文RAG**
- 输出逐题性能记录与汇总统计、并生成论文可用 Markdown 报告

重要参数：
- `--encrypted_collection`：必须指向一个**非空**的加密 collection（例如你现有的 `encrypted_documents_lihua`）
- `--plaintext_collection`：默认 `plaintext_documents_lihua_world`

（PowerShell 示例）
```powershell
$env:PYTHONUTF8=1
python D:\PycharmProjects\Graduation-project\unencrypted\bench\run_perf_compare_60q.py --encrypted_collection encrypted_documents_lihua
```

产物目录：`unencrypted/results_60q/<run_id>/`
- `run_meta.json`
- `samples.jsonl`（逐题记录，便于统计分析）
- `samples.csv`
- `summary.json`
- `REPORT.md`（可直接用于论文附录/实验章节）

> 脚本支持 `--limit` 做快速冒烟测试，例如 `--limit 2`。

---

## 5. 你遇到的报错解释与处理

### 5.1 `Vector database collection is empty`
含义：脚本默认使用 `config.yaml` 的 `vector_db.collection_name: encrypted_documents`，但该 collection 在你的本地是 0 points。

处理办法：
- **不要**新建额外数据库服务；你已经有本地 Qdrant storage。
- 直接让性能脚本指向你已有的非空加密 collection，例如：
  - `encrypted_documents_lihua`
  - `encrypted_documents_test1`

即运行：
```powershell
python ...\run_perf_compare_60q.py --encrypted_collection encrypted_documents_lihua
```

### 5.2 `Local Qdrant storage appears locked...`
Windows 下本地 Qdrant 的 `.lock` 文件可能残留/被占用。

处理建议：
- 确认没有其他脚本同时使用同一个 `qdrant_storage` 目录
- 明文实验与加密实验使用**不同 storage_path** 已大幅降低互锁概率

---

## 6. 结论报告怎么写（论文写作模板）

你可以在 `unencrypted/results_60q/<run_id>/REPORT.md` 基础上直接写“实验结论”段落，建议结构：

1. **实验设置**：说明仅改变 payload 加密/明文存储，其余模块一致，保证公平。
2. **总体延迟对比**：用 `total_time_s` 的 median/p95 对比，给出加密开销倍数。
3. **检索阶段开销**：重点解释加密带来的影响主要落在 retrieval + context build（解密/拼接）。
4. **生成阶段对比**：通常差异不大，说明 LLM 固有生成耗时主导。
5. **存储与资源**：对比 Qdrant 目录大小（payload 膨胀）与 RSS 峰值（解密/上下文组装）。
6. **结论**：在安全收益（明文泄露风险降低）与性能成本之间的权衡；结合你的系统应用场景给出选择建议。

---

## 7. 当前状态与下一步

- 明文索引：你已经成功执行并写入 `unencrypted/qdrant_storage_plaintext`，points=2242。
- 60Q 性能对比脚本：已提供 `unencrypted/bench/run_perf_compare_60q.py`，并在中断时也会尽量写出部分产物。

下一步建议：
1. 确认加密 collection 选择：推荐用 `encrypted_documents_lihua`（非空）。
2. 运行一次完整 60Q 对比（可能耗时较长，取决于 Ollama 模型生成速度）。
3. 将 `REPORT.md` 内容复制到论文的“实验评估”章节（本文件可作为实验方法与复现说明）。


