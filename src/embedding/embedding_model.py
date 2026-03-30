"""
Embedding Model
Lightweight embedding models for text vectorization
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import hashlib

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
        self.fallback = False

        logger.info(f"Loading embedding model: {model_name}")

        # Strategy:
        # 1) Try to load model online (may download into cache). This gives the best chance
        #    to obtain real SentenceTransformer embeddings when network is available.
        # 2) If online load fails (network or other), attempt to load from local cache.
        # 3) If both fail, fall back to deterministic hash embedding.
        try:
            # Try local cache first to avoid long download/retry delays when offline.
            try:
                self.model = SentenceTransformer(model_name, device=device, local_files_only=True)
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded from local cache. Dimension: {self.dimension}")
            except Exception as e_local:
                logger.info(f"Local cache load failed for {model_name}: {e_local}. Attempting online load...")
                try:
                    # Only try online load if local cache is not present; may still fail if network unreachable
                    self.model = SentenceTransformer(model_name, device=device, local_files_only=False)
                    self.dimension = self.model.get_sentence_embedding_dimension()
                    logger.info(f"Model loaded (online). Dimension: {self.dimension}")
                except Exception as e_online:
                    logger.warning(f"Online load failed for {model_name}: {e_online}.")
                    raise
        except Exception as e:
            # Final fallback: deterministic hash embedding for fully offline/unavailable model cases.
            logger.warning(f"Failed to load model {model_name} (local/online): {e}. Falling back to local hash embedding.")
            self.model = None
            self.dimension = 384
            self.fallback = True

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
            if not self.fallback and self.model is not None:
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    normalize_embeddings=normalize,
                    convert_to_numpy=True
                )
                return embeddings

            # Fallback deterministic hash embedding (cheap, no external deps)
            embs = []
            for t in texts:
                if t is None:
                    t = ''
                # Tokenize simply by whitespace; if no spaces, fall back to characters
                tokens = t.split() if len(t.split())>1 else list(t)
                vec = np.zeros(self.dimension, dtype=float)
                for tok in tokens:
                    # create a 32-byte digest
                    h = hashlib.sha256(tok.encode('utf-8')).digest()
                    # add digest bytes into the 384-d vector by repeating
                    for i in range(self.dimension):
                        vec[i] += h[i % len(h)]
                # normalize to unit length if requested
                if normalize:
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                embs.append(vec)
            return np.vstack(embs)

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
