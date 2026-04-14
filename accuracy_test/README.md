# Accuracy Test (RAG)

This folder contains a **standalone** accuracy evaluation experiment for the project's RAG system.
It does **not** modify the existing RAG pipeline; it only calls it.

## Datasets

We evaluate three 60-question test sets:

- **Multi**: `data/test_datasets/lihua-queries1` with gold answers in `data/gold-answer/lihua-queries1-gold-answer`
- **Single**: `data/test_datasets/lihua-queries2` with gold answers in `data/gold-answer/lihua-queries2-gold-answer`
- **Null**: `data/test_datasets/lihua-queries3` with gold answers in `data/gold-answer/lihua-queries3-gold-answer`

## Before you run (common failure)

### 1) `embeddings.position_ids UNEXPECTED`

This message comes from the sentence-transformers model loader. It's typically harmless and can be ignored.

### 2) `Vector database collection is empty`

Your default collection is controlled by `config/config.yaml`:

- `vector_db.collection_name: encrypted_documents`

If that collection has **0 points**, the RAG pipeline will stop with:

- `RuntimeError: Vector database collection is empty ... has 0 points`

Fix options:

- **Use a non-empty existing collection** (recommended):
  - Example (use one of the names printed in the error message):

```powershell
python -m accuracy_test.run_rag_accuracy_eval --config config/config.yaml --key_file encryption.key --collection_name encrypted_documents_lihua
```

- **Ingest documents into your target collection** (also fine):

```powershell
python -m scripts.ingest_documents --input_dir data\single_test1 --collection_name encrypted_documents
```

- **Debug only**: run with empty collection allowed (results are not meaningful):

```powershell
python -m accuracy_test.run_rag_accuracy_eval --allow_empty_collection
```

## Run

From repo root:

```powershell
python -m accuracy_test.run_rag_accuracy_eval --config config/config.yaml --key_file encryption.key
```

Optional flags:

```powershell
# quick smoke test, 3 questions per dataset
python -m accuracy_test.run_rag_accuracy_eval --limit 3

# slow down a bit to avoid overloading local LLM
python -m accuracy_test.run_rag_accuracy_eval --sleep 0.2
```

## Outputs

Each run writes a timestamped directory:

- `accuracy_test/runs/<timestamp>_<llm>_<collection>/predictions.jsonl`
- `accuracy_test/runs/<timestamp>_<llm>_<collection>/per_question.csv`
- `accuracy_test/runs/<timestamp>_<llm>_<collection>/per_question.json`
- `accuracy_test/runs/<timestamp>_<llm>_<collection>/summary.json`
- `accuracy_test/runs/<timestamp>_<llm>_<collection>/report.md`

## Metric definitions (short)

- **Single**: token-overlap Precision/Recall/F1 between prediction and gold.
- **Multi**: gold is a set of items (split by `&` etc); we treat an item as *predicted* if it appears in the prediction string.
- **Null**: gold is unanswerable; success means abstaining (`I don't know` / `Insufficient information` etc).
- **Overall**: micro-average across all questions using summed TP/FP/FN.

