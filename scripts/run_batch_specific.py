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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--queries', type=str, required=True, help='Path to queries file')
parser.add_argument('--output', type=str, default=None, help='Output jsonl file')
args = parser.parse_args()

CONFIG_PATH = 'config/config.yaml'
KEY_FILE = 'encryption.key'
QUERIES_FILE = args.queries
OUTPUT_DIR = 'results'
if args.output:
    OUTPUT_FILE = args.output
else:
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f'batch_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl')

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Load encryption
enc = AESEncryption(key_size=config['encryption']['key_size'])
enc.load_key(KEY_FILE)
print('Loaded encryption key')

# Embedding
em = EmbeddingModel(model_name=config['embedding']['model_name'])
print('Loaded embedding model, dim=', em.get_dimension())

# Vector store
vs = VectorStore(
    collection_name=config['vector_db']['collection_name'],
    dimension=em.get_dimension(),
    distance_metric=config['vector_db']['distance_metric'],
    storage_path=config['vector_db']['storage_path']
)
print('Connected to VectorStore')

retriever = Retriever(em, vs, enc)

# LLM client (may be unavailable but we still proceed)
llm_client = OllamaClient(base_url=config['llm']['base_url'], model_name=config['llm']['model_name'])

rag = RAGSystem(retriever=retriever, llm_client=llm_client, prompt_template=config['rag']['prompt_template'], max_context_length=config['rag']['max_context_length'])

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
            print('Answer:', (res.get('answer') or '')[:200])
            print('Retrieved chunks:', res.get('num_chunks_retrieved'))
        except Exception as e:
            print('Error for query:', q, e)
            out.write(json.dumps({'query': q, 'error': str(e)}) + '\n')

print('\nBatch run complete. Results saved to', OUTPUT_FILE)

