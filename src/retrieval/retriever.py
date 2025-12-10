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
        
        for result in search_results:
            try:
                # Decrypt the chunk
                plaintext = self.encryption.decrypt(
                    result['ciphertext'],
                    result['nonce']
                )
                
                decrypted_result = {
                    'text': plaintext,
                    'score': result['score'],
                    'metadata': result['metadata']
                }
                
                # Optionally include encrypted data
                if return_encrypted:
                    decrypted_result['ciphertext'] = result['ciphertext']
                    decrypted_result['nonce'] = result['nonce']
                
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
