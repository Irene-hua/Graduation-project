import os, sys, yaml, json
sys.path.insert(0, os.path.abspath('..'))
import sys
sys.path.insert(0, os.path.abspath('.'))
from src.encryption import AESEncryption
from src.embedding import EmbeddingModel
from src.retrieval import VectorStore, Retriever
from src.llm.ollama import OllamaClient
from src.rag_pipeline.rag_system import RAGSystem

CONFIG_PATH = 'config/config.yaml'
QUERIES_FILE = 'data/test_datasets/Lihua-World-queries'

cfg = yaml.safe_load(open(CONFIG_PATH,'r',encoding='utf-8'))
enc = AESEncryption(key_size=cfg['encryption']['key_size']); enc.load_key('encryption.key')
em = EmbeddingModel(model_name=cfg['embedding']['model_name'])
vs = VectorStore(collection_name=cfg['vector_db']['collection_name'], dimension=em.get_dimension(), distance_metric=cfg['vector_db']['distance_metric'], host=cfg['vector_db']['host'], port=cfg['vector_db']['port'])
retriever = Retriever(em, vs, enc)
llm_name = cfg.get('llm', {}).get('default_model') or cfg.get('llm', {}).get('model_name', 'mistral')
llm = OllamaClient(base_url=cfg['llm']['base_url'], model_name=llm_name)
print('Using LLM model:', llm_name)
rag = RAGSystem(retriever=retriever, llm_client=llm, prompt_template=None, max_context_length=cfg['rag']['max_context_length'])

queries = [line.strip() for line in open(QUERIES_FILE,'r',encoding='utf-8') if line.strip()]
results=[]
for q in queries:
    print('Query:', q)
    try:
        res = rag.answer_question(q, top_k=None, temperature=0.0, max_tokens=64)
        ans = res.get('answer')
        # flatten newline
        ans = (ans or '').replace('\n',' ').strip()
        print('Answer:', ans)
        results.append({'query':q,'answer':ans,'meta':{'num_chunks':res.get('num_chunks_retrieved')}})
    except Exception as e:
        print('Error processing query:', e)
        results.append({'query':q,'error':str(e)})

# write output file
outf = os.path.join('results', 'deterministic_batch_results.jsonl')
with open(outf,'w',encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False)+'\n')

print('\nDone. Results written to',outf)
