import os
import sys
import yaml
import json
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.encryption import AESEncryption
from src.embedding import EmbeddingModel
from src.retrieval import VectorStore, Retriever
from src.llm.ollama_client import OllamaClient
from src.rag_pipeline.rag_system import RAGSystem

CONFIG_PATH = 'config/config.yaml'
KEY_FILE = 'encryption.key'
QUERIES_FILE = 'data/test_datasets/Lihua-World-queries'
OUTPUT_DIR = 'results'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f'batch_results_chinese_prompt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl')

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Load encryption key
enc = AESEncryption(key_size=config['encryption']['key_size'])
enc.load_key(KEY_FILE)
print('Loaded encryption key')

# Embedding model
em = EmbeddingModel(model_name=config['embedding']['model_name'])
print('Loaded embedding model, dim=', em.get_dimension())

# Vector store and retriever
vs = VectorStore(
    collection_name=config['vector_db']['collection_name'],
    dimension=em.get_dimension(),
    distance_metric=config['vector_db']['distance_metric'],
    storage_path=config['vector_db']['storage_path']
)
retriever = Retriever(em, vs, enc)

# LLM client
llm_client = OllamaClient(base_url=config['llm']['base_url'], model_name=config['llm']['model_name'])

# Chinese prompt template (override)
chinese_prompt = """请根据下面的上下文（来自本地文档），用中文简洁、准确地回答问题。

上下文：
{context}

问题：{question}

要求：
- 用中文回答，直接给出结论。
- 如果上下文中没有明确答案，请说明“在提供的文档中未找到明确答案”。
- 如果可能，请在回答末尾列出引用的来源文件名和 chunk_id。

回答："""

rag = RAGSystem(retriever=retriever, llm_client=llm_client, prompt_template=chinese_prompt, max_context_length=config['rag']['max_context_length'])

# Read queries
with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
    queries = [line.strip() for line in f if line.strip()]

print(f'Read {len(queries)} queries from {QUERIES_FILE}')

with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
    for q in queries:
        print('\n=== Query:', q)
        try:
            res = rag.answer_question(q, top_k=config['retrieval'].get('default_top_k', 5), temperature=config['llm'].get('temperature', 0.7), max_tokens=config['llm'].get('max_tokens'))
            out.write(json.dumps({'query': q, 'answer': res.get('answer'), 'num_chunks': res.get('num_chunks_retrieved'), 'used_chunks': res.get('used_chunks')}, ensure_ascii=False) + '\n')
            print('Answer snippet:', (res.get('answer') or '')[:200])
            print('Retrieved chunks:', res.get('num_chunks_retrieved'))
        except Exception as e:
            print('Error for query:', q, e)
            out.write(json.dumps({'query': q, 'error': str(e)}) + '\n')

print('\nBatch run complete. Results saved to', OUTPUT_FILE)

