"""
Embedding Model
Lightweight embedding models for text vectorization
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for sentence transformer embedding models"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                 device: str = None):
        """
        Initialize embedding model
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        
        logger.info(f"Loading embedding model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Model loaded successfully. Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], 
               batch_size: int = 32,
               show_progress: bool = False,
               normalize: bool = True) -> np.ndarray:
        """
        Encode text(s) into embeddings
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings to unit length
            
        Returns:
            Numpy array of embeddings (shape: [n_texts, dimension])
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise
    
    def encode_queries(self, queries: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode queries (alias for encode, for clarity)
        
        Args:
            queries: Query text(s)
            **kwargs: Additional arguments for encode
            
        Returns:
            Query embeddings
        """
        return self.encode(queries, **kwargs)
    
    def encode_documents(self, documents: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode documents (alias for encode, for clarity)
        
        Args:
            documents: Document text(s)
            **kwargs: Additional arguments for encode
            
        Returns:
            Document embeddings
        """
        return self.encode(documents, **kwargs)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension
    
    def get_model_name(self) -> str:
        """Get model name"""
        return self.model_name
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    @staticmethod
    def batch_cosine_similarity(query_vec: np.ndarray, 
                                doc_vecs: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and multiple documents
        
        Args:
            query_vec: Query vector (1D array)
            doc_vecs: Document vectors (2D array, shape: [n_docs, dimension])
            
        Returns:
            Array of similarity scores
        """
        # Normalize vectors
        query_norm = query_vec / np.linalg.norm(query_vec)
        doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        
        # Calculate dot product
        similarities = np.dot(doc_norms, query_norm)
        
        return similarities
