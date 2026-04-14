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
from src.rag_pipeline import RAGSystem, LocalReranker

parser = argparse.ArgumentParser(description='Run batch Chinese prompt evaluation')
parser.add_argument('--config', type=str, default='config/config.yaml')
parser.add_argument('--key_file', type=str, default='encryption.key')
parser.add_argument('--queries_file', type=str, default='data/test_datasets/Lihua-World-queries')
parser.add_argument('--collection_name', type=str, default=None, help='Override Qdrant collection name for this batch run')
args = parser.parse_args()

CONFIG_PATH = args.config
KEY_FILE = args.key_file
QUERIES_FILE = args.queries_file
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
collection_name = args.collection_name or config['vector_db']['collection_name']
vs = VectorStore(
    collection_name=collection_name,
    dimension=em.get_dimension(),
    distance_metric=config['vector_db']['distance_metric'],
    storage_path=config['vector_db']['storage_path'],
    host=config['vector_db'].get('host'),
    port=config['vector_db'].get('port')
)
print('Using collection:', collection_name)
print('Vector storage path:', config['vector_db']['storage_path'])
retriever = Retriever(em, vs, enc)
print('Using local-first vector store and retriever')

# LLM client
llm_name = config.get('llm', {}).get('default_model') or config.get('llm', {}).get('model_name', 'mistral')
llm_client = OllamaClient(base_url=config['llm']['base_url'], model_name=llm_name)
print('Using LLM model:', llm_name)

# Prompt template: answer in the same language as the question and be concise; do NOT list evidence
prompt_template = """Please answer the question using only the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer in the same language the question is written.
- Give a concise direct conclusion (e.g., Yes/No or short factual answer) and one short sentence explanation if needed.
- Do NOT list or print evidence sources or citations in the answer.
- If the provided context does not contain a clear answer, reply: "In the provided documents there is no clear answer." (use the question language if possible).

Answer:"""

# Optional local reranker (two-stage retrieval)
reranker = None
if config.get('rerank', {}).get('enabled', False):
    reranker = LocalReranker(
        max_candidates=config['rerank'].get('max_candidates', 20),
        min_score=config['rerank'].get('min_score', 0.0)
    )
    print('Local reranker enabled')

# Wrap main RAG execution in try/finally so we explicitly close the Qdrant client (avoid relying on destructor at interpreter shutdown)
try:
    rag = RAGSystem(retriever=retriever, llm_client=llm_client, prompt_template=prompt_template, max_context_length=config['rag']['max_context_length'], reranker=reranker)
    print('Using local-first RAG pipeline (retrieve → rerank → context → generate)')

    # Read queries
    with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]

    print(f'Read {len(queries)} queries from {QUERIES_FILE}')

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        for q in queries:
            print('\n=== Query:', q)
            try:
                # Use a finite top_k to keep batch runs stable.
                # For temporal/date questions, enable local fallback so header-style chunks can be recovered.
                top_k = config['retrieval'].get('default_top_k', 20)
                print(f'Invoking RAG.answer_question (top_k={top_k})')
                res = rag.answer_question(
                    q,
                    top_k=top_k,
                    temperature=config['llm'].get('temperature', 0.7),
                    max_tokens=config['llm'].get('max_tokens'),
                )
                out.write(json.dumps({'query': q, 'answer': res.get('answer'), 'num_chunks': int(res.get('num_chunks_retrieved') or 0), 'used_chunks': res.get('used_chunks') or [], 'rerank_enabled': res.get('rerank_enabled', False), 'rerank_before_top1': res.get('rerank_before_top1'), 'rerank_after_top1': res.get('rerank_after_top1'), 'rerank_top_scores': res.get('rerank_top_scores') or [], 'context_length': res.get('context_length'), 'retrieve_k': res.get('retrieve_k'), 'weak_answer': res.get('weak_answer', False)}, ensure_ascii=False) + '\n')
                print('Answer snippet:', (res.get('answer') or '')[:200])
                print('Retrieved chunks:', res.get('num_chunks_retrieved'))
                print('Retrieve K:', res.get('retrieve_k'))
                print('Context length:', res.get('context_length'))
                print('Weak answer:', res.get('weak_answer', False))
                if res.get('rerank_enabled'):
                    print('Rerank enabled:', res.get('rerank_enabled'))
                    print('Rerank before top1:', res.get('rerank_before_top1') or 'n/a')
                    print('Rerank after top1:', res.get('rerank_after_top1') or 'n/a')
                    print('Rerank top scores:', res.get('rerank_top_scores') or [])
            except Exception as e:
                print('Error for query:', q, e)
                out.write(json.dumps({'query': q, 'error': str(e), 'num_chunks': 0, 'used_chunks': []}, ensure_ascii=False) + '\n')

    print('\nBatch run complete. Results saved to', OUTPUT_FILE)

except Exception as main_e:
    print('Batch run failed with exception:', main_e)

finally:
    # Explicitly close the qdrant client to avoid relying on destructor during interpreter shutdown
    try:
        if hasattr(vs, 'client') and vs.client is not None:
            try:
                vs.client.close()
                print('Closed Qdrant client')
            except Exception:
                # Ignore exceptions during close; they often happen during interpreter teardown
                pass
    except Exception:
        pass

    # Close llm client if it exposes a close method
    try:
        if hasattr(llm_client, 'close'):
            try:
                llm_client.close()
                print('Closed LLM client')
            except Exception:
                pass
    except Exception:
        pass
