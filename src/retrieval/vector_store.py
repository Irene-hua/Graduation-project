"""
Vector Store using Qdrant
Stores encrypted chunks with their vector representations
"""

from typing import List, Dict, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue
)
import uuid
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """Qdrant vector database wrapper"""
    
    def __init__(self, 
                 collection_name: str = 'encrypted_documents',
                 dimension: int = 384,
                 distance_metric: str = 'Cosine',
                 storage_path: str = './qdrant_storage',
                 host: Optional[str] = None,
                 port: Optional[int] = None):
        """
        Initialize Qdrant vector store
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            distance_metric: Distance metric ('Cosine', 'Euclidean', 'Dot')
            storage_path: Path for local storage (used if host is None)
            host: Qdrant server host (None for local storage)
            port: Qdrant server port
        """
        self.collection_name = collection_name
        self.dimension = dimension
        
        # Map string to Distance enum
        distance_map = {
            'Cosine': Distance.COSINE,
            'Euclidean': Distance.EUCLID,
            'Dot': Distance.DOT
        }
        self.distance = distance_map.get(distance_metric, Distance.COSINE)
        
        # Initialize client
        if host:
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"Connected to Qdrant server at {host}:{port}")
        else:
            self.client = QdrantClient(path=storage_path)
            logger.info(f"Using local Qdrant storage at {storage_path}")
        
        # Create collection if it doesn't exist
        self._create_collection_if_not_exists()
    
    def _create_collection_if_not_exists(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=self.distance
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def add_vectors(self, 
                    vectors: np.ndarray,
                    encrypted_chunks: List[Dict],
                    metadata: List[Dict] = None) -> List[str]:
        """
        Add vectors with encrypted chunks to database
        
        Args:
            vectors: Numpy array of vectors (shape: [n, dimension])
            encrypted_chunks: List of dicts with 'ciphertext' and 'nonce'
            metadata: Optional additional metadata for each vector
            
        Returns:
            List of assigned point IDs
        """
        if len(vectors) != len(encrypted_chunks):
            raise ValueError("Number of vectors must match number of encrypted chunks")
        
        if metadata and len(metadata) != len(vectors):
            raise ValueError("Number of metadata items must match number of vectors")
        
        points = []
        point_ids = []
        
        for i, (vector, encrypted_chunk) in enumerate(vectors, encrypted_chunks):
            # Generate unique ID
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            # Prepare payload
            payload = {
                'ciphertext': encrypted_chunk['ciphertext'],
                'nonce': encrypted_chunk['nonce'],
                'chunk_id': encrypted_chunk.get('chunk_id', i)
            }
            
            # Add metadata if provided
            if metadata and i < len(metadata):
                payload.update(metadata[i])
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload
            )
            points.append(point)
        
        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        logger.info(f"Added {len(points)} vectors to collection")
        return point_ids
    
    def search(self, 
               query_vector: np.ndarray,
               top_k: int = 5,
               filter_dict: Dict = None) -> List[Dict]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filter_dict: Optional filter conditions
            
        Returns:
            List of search results with scores and payloads
        """
        # Prepare filter if provided
        query_filter = None
        if filter_dict:
            # Convert filter_dict to Qdrant filter
            # This is a simple implementation - extend as needed
            pass
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result.id,
                'score': result.score,
                'ciphertext': result.payload['ciphertext'],
                'nonce': result.payload['nonce'],
                'metadata': {k: v for k, v in result.payload.items() 
                           if k not in ['ciphertext', 'nonce']}
            })
        
        return formatted_results
    
    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(collection_name=self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            'name': self.collection_name,
            'vectors_count': info.vectors_count,
            'points_count': info.points_count,
            'status': info.status
        }
    
    def count(self) -> int:
        """Get number of vectors in collection"""
        info = self.client.get_collection(collection_name=self.collection_name)
        return info.points_count
