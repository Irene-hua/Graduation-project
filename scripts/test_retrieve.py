import os
import yaml
import logging
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.encryption import AESEncryption
from src.embedding import EmbeddingModel
from src.retrieval import VectorStore, Retriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_retrieve')

CONFIG_PATH = 'config/config.yaml'
KEY_FILE = 'encryption.key'

def main():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Load encryption
    enc = AESEncryption(key_size=config['encryption']['key_size'])
    if not os.path.exists(KEY_FILE):
        logger.error('encryption.key not found')
        return
    enc.load_key(KEY_FILE)
    logger.info('Loaded encryption key')

    # Embedding
    em = EmbeddingModel(model_name=config['embedding']['model_name'])
    logger.info(f'Embedding dimension: {em.get_dimension()}')

    # Vector store
    vs = VectorStore(
        collection_name=config['vector_db']['collection_name'],
        dimension=em.get_dimension(),
        distance_metric=config['vector_db']['distance_metric'],
        storage_path=config['vector_db']['storage_path']
    )
    logger.info('VectorStore ready')

    retriever = Retriever(em, vs, enc)

    query = "Planned improvements包含哪些？"
    logger.info(f'Query: {query}')

    # Call internal vector store search directly too
    qv = em.encode(query)
    raw_results = vs.search(qv, top_k=5)
    logger.info(f'Raw search results (formatted): {raw_results}')

    # Retrieve decrypted
    dec_results = retriever.retrieve(query, top_k=5)
    logger.info(f'Decrypted results: {dec_results}')

    for i, r in enumerate(dec_results):
        print('\n--- Decrypted chunk', i)
        print('text:', r.get('text'))
        print('score:', r.get('score'))
        print('metadata:', r.get('metadata'))

if __name__ == '__main__':
    main()
