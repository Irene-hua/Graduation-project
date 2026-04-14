import os
import sys
import yaml
import json
from datetime import datetime
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.encryption import AESEncryption
from src.embedding import EmbeddingModel
from src.retrieval import VectorStore, Retriever
from src.llm.ollama_client import OllamaClient
from src.rag_pipeline.rag_system import RAGSystem

parser = argparse.ArgumentParser(description='Run batch queries against a selected collection')
parser.add_argument('--config', type=str, default='config/config.yaml')
parser.add_argument('--key_file', type=str, default='encryption.key')
parser.add_argument('--queries_file', type=str, default='data/test_datasets/test_queries.txt')
parser.add_argument('--collection_name', type=str, default=None, help='Override Qdrant collection name for this batch run')
args = parser.parse_args()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH = args.config
if not os.path.isabs(CONFIG_PATH):
    CONFIG_PATH = os.path.join(PROJECT_ROOT, CONFIG_PATH)
KEY_FILE = args.key_file
if not os.path.isabs(KEY_FILE):
    KEY_FILE = os.path.join(PROJECT_ROOT, KEY_FILE)
QUERIES_FILE = args.queries_file
if not os.path.isabs(QUERIES_FILE):
    QUERIES_FILE = os.path.join(PROJECT_ROOT, QUERIES_FILE)
OUTPUT_DIR = 'results'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f'batch_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl')

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Load encryption
enc = AESEncryption(key_size=config['encryption']['key_size'])
enc.load_key(KEY_FILE)
print('Loaded encryption key')

# Embedding
em = EmbeddingModel(model_name=config['embedding']['model_name'])
print('Loaded embedding model, dim=', em.get_dimension())

# Vector store
collection_name = args.collection_name or config['vector_db']['collection_name']
vs = VectorStore(
    collection_name=collection_name,
    dimension=em.get_dimension(),
    distance_metric=config['vector_db']['distance_metric'],
    storage_path=config['vector_db']['storage_path']
)
print('Connected to VectorStore')
print('Using collection:', collection_name)

retriever = Retriever(em, vs, enc)

# LLM client (we'll still try to initialize Ollama client but RAGSystem can work if model not available for testing retrieval)
llm_name = config.get('llm', {}).get('default_model') or config.get('llm', {}).get('model_name', 'mistral')
llm_client = OllamaClient(base_url=config['llm']['base_url'], model_name=llm_name)
print('Using LLM model:', llm_name)

rag = RAGSystem(retriever=retriever, llm_client=llm_client, prompt_template=config['rag']['prompt_template'], max_context_length=config['rag']['max_context_length'])

# Read queries
with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
    queries = [line.strip() for line in f if line.strip()]

print(f'Read {len(queries)} queries')

with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
    for q in queries:
        print('\n=== Query:', q)
        try:
            res = rag.answer_question(q, top_k=config['retrieval'].get('default_top_k', 5), temperature=config['llm'].get('temperature', 0.7), max_tokens=config['llm'].get('max_tokens'))
            # Write compact JSON line
            out.write(json.dumps({'query': q, 'answer': res.get('answer'), 'num_chunks': res.get('num_chunks_retrieved'), 'used_chunks': res.get('used_chunks')}, ensure_ascii=False) + '\n')
            print('Answer:', res.get('answer')[:200])
            print('Retrieved chunks:', res.get('num_chunks_retrieved'))
        except Exception as e:
            print('Error for query:', q, e)
            out.write(json.dumps({'query': q, 'error': str(e)}) + '\n')

print('\nBatch run complete. Results saved to', OUTPUT_FILE)
