import os, sys, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.encryption import AESEncryption
from src.embedding import EmbeddingModel
from src.retrieval import VectorStore, Retriever

CONFIG='config/config.yaml'
KEY='encryption.key'

with open(CONFIG,'r') as f:
    cfg = yaml.safe_load(f)

enc = AESEncryption(key_size=cfg['encryption']['key_size'])
enc.load_key(KEY)
print('Loaded encryption key')

em = EmbeddingModel(model_name=cfg['embedding']['model_name'])
print('Loaded embedding model, dim=', em.get_dimension())

vs = VectorStore(collection_name=cfg['vector_db']['collection_name'], dimension=em.get_dimension(), distance_metric=cfg['vector_db']['distance_metric'], storage_path=cfg['vector_db']['storage_path'])
print('Connected to vector store')

retriever = Retriever(em, vs, enc)
q = 'What is machine learning?'
print('Query:', q)
res = retriever.retrieve(q, top_k=5)
print('Results count:', len(res))
for i,r in enumerate(res):
    print('---', i)
    print('text:', r.get('text')[:200])
    print('score:', r.get('score'))
    print('metadata:', r.get('metadata'))

