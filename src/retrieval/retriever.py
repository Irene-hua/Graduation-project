"""
Retriever
Combines embedding, vector search, and decryption for retrieval
"""

from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Retriever:
    """High-level retriever combining embedding, search, and decryption"""
    
    def __init__(self, embedding_model, vector_store, encryption):
        """
        Initialize retriever
        
        Args:
            embedding_model: EmbeddingModel instance
            vector_store: VectorStore instance
            encryption: AESEncryption instance
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.encryption = encryption
    
    def retrieve(self, query: str, top_k: int = 5, 
                 return_encrypted: bool = False) -> List[Dict]:
        """
        Retrieve and decrypt relevant chunks for a query
        
        Args:
            query: Query text
            top_k: Number of results to retrieve
            return_encrypted: If True, also return encrypted text
            
        Returns:
            List of retrieved chunks with decrypted text and metadata
        """
        # Encode query
        logger.info(f"Encoding query: {query[:50]}...")
        query_vector = self.embedding_model.encode(query)
        
        # Search vector database
        logger.info(f"Searching for top-{top_k} similar chunks...")
        search_results = self.vector_store.search(query_vector, top_k=top_k)
        
        # Decrypt results
        logger.info(f"Decrypting {len(search_results)} chunks...")
        decrypted_results = []
        
        # Fallback: if search_results empty or payloads missing ciphertext/nonce, try local scroll+similarity
        need_fallback = False
        if not search_results:
            need_fallback = True
        else:
            # check first result for ciphertext presence
            first = search_results[0]
            if not first.get('ciphertext'):
                need_fallback = True

        if need_fallback:
            try:
                logger.info('Using local scroll fallback to obtain payloads and vectors')
                records = self.vector_store.client.scroll(collection_name=self.vector_store.collection_name, limit=10000, with_payload=True, with_vectors=True)
                recs = records[0] if isinstance(records, tuple) else records
                vecs = []
                metas = []
                for rec in recs:
                    v = getattr(rec, 'vector', None) or (rec.get('vector') if hasattr(rec, 'get') else None)
                    if v is None:
                        continue
                    try:
                        pv = getattr(rec, 'payload', None) or (rec.payload if hasattr(rec, 'payload') else {})
                        if hasattr(pv, 'to_dict'):
                            pd = pv.to_dict()
                        else:
                            pd = dict(pv) if pv is not None else {}
                    except Exception:
                        pd = {}
                    vecs.append(np.asarray(v))
                    metas.append({'id': getattr(rec, 'id', None), 'payload': pd, 'score': getattr(rec, 'score', None)})

                if vecs:
                    M = np.vstack(vecs)
                    qv = np.asarray(query_vector).reshape(-1)
                    try:
                        qn = qv / np.linalg.norm(qv)
                        Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
                        sims = Mn.dot(qn)
                    except Exception:
                        sims = M.dot(qv)
                    idxs = np.argsort(-sims)[:top_k]
                    search_results = []
                    for ix in idxs:
                        item = metas[ix]
                        pd = item['payload']
                        search_results.append({
                            'id': item.get('id'),
                            'score': float(sims[ix]),
                            'ciphertext': pd.get('ciphertext'),
                            'nonce': pd.get('nonce'),
                            'metadata': {k: v for k, v in pd.items() if k not in ['ciphertext', 'nonce']}
                        })
                    logger.info(f'Local fallback produced {len(search_results)} candidates')
            except Exception as e:
                logger.debug(f'Local scroll fallback failed: {e}')

        for result in search_results:
            try:
                # Locate ciphertext/nonce or fallback to plaintext stored in payload
                ciphertext = result.get('ciphertext')
                nonce = result.get('nonce')

                # Normalize metadata
                metadata = result.get('metadata') if result.get('metadata') is not None else {}
                if not isinstance(metadata, dict):
                    # leave as-is, but prefer dict where possible
                    try:
                        metadata = dict(metadata)
                    except Exception:
                        metadata = {}

                # Some clients may nest the encrypted fields in metadata
                if (ciphertext is None or nonce is None) and isinstance(metadata, dict):
                    # common fallback names
                    ciphertext = ciphertext or metadata.get('ciphertext') or metadata.get('ct')
                    nonce = nonce or metadata.get('nonce') or metadata.get('n')

                plaintext = None
                if ciphertext is None or nonce is None:
                    # Maybe the payload contains the plaintext directly under common keys
                    for key in ('text', 'plaintext', 'content', 'document', 'source_text'):
                        if key in metadata and metadata[key]:
                            plaintext = metadata[key]
                            break

                if plaintext is None:
                    # If we still have ciphertext/nonce, attempt decryption
                    if ciphertext is None or nonce is None:
                        raise ValueError('Missing ciphertext or nonce in search result payload')

                    # Decrypt (AESEncryption.decrypt expects base64 strings)
                    plaintext = self.encryption.decrypt(ciphertext, nonce)

                decrypted_result = {
                    'text': plaintext,
                    'score': result.get('score'),
                    'metadata': metadata
                }

                # Optionally include encrypted data
                if return_encrypted:
                    decrypted_result['ciphertext'] = ciphertext
                    decrypted_result['nonce'] = nonce

                decrypted_results.append(decrypted_result)

            except Exception as e:
                logger.error(f"Failed to decrypt chunk {result.get('id', 'unknown')}: {e}")

        logger.info(f"Successfully retrieved and decrypted {len(decrypted_results)} chunks")
        return decrypted_results
    
    def retrieve_batch(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """
        Retrieve for multiple queries
        
        Args:
            queries: List of query texts
            top_k: Number of results per query
            
        Returns:
            List of result lists, one per query
        """
        results = []
        for query in queries:
            query_results = self.retrieve(query, top_k=top_k)
            results.append(query_results)
        return results
    
    def evaluate_retrieval(self, 
                          query: str,
                          ground_truth_ids: List[str],
                          top_k: int = 5) -> Dict:
        """
        Evaluate retrieval performance for a query
        
        Args:
            query: Query text
            ground_truth_ids: List of relevant chunk IDs
            top_k: Number of results to retrieve
            
        Returns:
            Dict with precision, recall, and F1 metrics
        """
        # Retrieve results
        results = self.retrieve(query, top_k=top_k)
        
        # Extract retrieved IDs
        retrieved_ids = [r['metadata'].get('chunk_id', '') for r in results]
        
        # Calculate metrics
        relevant_retrieved = len(set(retrieved_ids) & set(ground_truth_ids))
        
        precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0
        recall = relevant_retrieved / len(ground_truth_ids) if ground_truth_ids else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'retrieved_count': len(retrieved_ids),
            'relevant_count': len(ground_truth_ids),
            'relevant_retrieved': relevant_retrieved
        }
