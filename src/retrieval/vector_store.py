"""
Vector Store using Qdrant
Stores encrypted chunks with their vector representations
"""

from typing import List, Dict, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct
)
import uuid
import logging
import inspect

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
        
        for i, (vector, encrypted_chunk) in enumerate(zip(vectors, encrypted_chunks)):
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
        # Normalize and validate query_vector
        try:
            # Allow callers to pass list or numpy array
            if isinstance(query_vector, list):
                query_vec = np.array(query_vector)
            else:
                query_vec = np.asarray(query_vector)

            # If shape is (1, dim) extract the single vector
            if query_vec.ndim == 2 and query_vec.shape[0] == 1:
                query_vec = query_vec[0]

            # Ensure we have a 1D vector
            if query_vec.ndim != 1:
                raise ValueError(f"query_vector must be 1D or shape (1, dim). Got ndim={query_vec.ndim}")

            # Validate dimension
            if query_vec.shape[0] != self.dimension:
                raise ValueError(f"Query vector dimension ({query_vec.shape[0]}) does not match store dimension ({self.dimension})")

            query_list = query_vec.tolist()
        except Exception as e:
            logger.error(f"Invalid query vector provided: {e}")
            # Return empty list so caller can handle lack of results gracefully
            return []

        # Prepare filter if provided
        query_filter = None
        if filter_dict:
            # Note: Filter implementation can be extended based on specific needs
            # For now, filters are not supported - this is a future enhancement
            logger.debug("Filters supplied to VectorStore.search but filtering is not implemented")
            raise NotImplementedError("Filtering is not yet implemented. This is a future feature.")

        # Search
        try:
            # If local scroll is available, prefer it for local storage to ensure payloads are returned
            if hasattr(self.client, 'scroll'):
                try:
                    logger.info('Using local scroll + local similarity for search (preferred for local storage)')
                    records = self.client.scroll(collection_name=self.collection_name, limit=10000, with_payload=True, with_vectors=True)
                    recs = records[0] if isinstance(records, tuple) else records

                    vecs = []
                    rec_meta = []
                    for rec in recs:
                        vec = getattr(rec, 'vector', None) or (rec.get('vector') if hasattr(rec, 'get') else None)
                        if vec is None:
                            continue
                        v = np.asarray(vec)
                        payload = getattr(rec, 'payload', None) or (rec.payload if hasattr(rec, 'payload') else {})
                        try:
                            if hasattr(payload, 'to_dict'):
                                pd = payload.to_dict()
                            else:
                                pd = dict(payload) if payload is not None else {}
                        except Exception:
                            pd = {}
                        vecs.append(v)
                        rec_meta.append((getattr(rec, 'id', None), pd, getattr(rec, 'score', None)))

                    if vecs:
                        mat = np.vstack(vecs)
                        qv = np.asarray(query_list)
                        try:
                            qv_norm = qv / np.linalg.norm(qv)
                            mat_norms = mat / np.linalg.norm(mat, axis=1, keepdims=True)
                            sims = mat_norms.dot(qv_norm)
                        except Exception:
                            sims = np.dot(mat, qv)

                        idxs = np.argsort(-sims)[:top_k]
                        formatted_results = []
                        for ix in idxs:
                            pid, pd, sc = rec_meta[ix]
                            formatted_results.append({
                                'id': pid,
                                'score': float(sims[ix]) if sims is not None else sc,
                                'ciphertext': pd.get('ciphertext'),
                                'nonce': pd.get('nonce'),
                                'metadata': {k: v for k, v in pd.items() if k not in ['ciphertext', 'nonce']}
                            })
                        logger.info(f'Local scroll returned {len(formatted_results)} results')
                        return formatted_results
                except Exception as e:
                    logger.debug(f'Local scroll approach failed or not suitable: {e}')

            # Try several possible search method names/signatures to support different qdrant-client versions
            results = None
            attempted_methods = []

            candidate_names = [
                'search', 'search_points', 'search_collection', 'search_batch', 'search_by_vector',
                'query', 'query_points', 'query_batch', 'query_points_groups',
                'retrieve'
            ]

            # A pool of possible kwargs we can supply; we'll filter by method signature
            possible_kwargs = {
                'collection_name': self.collection_name,
                'query_vector': query_list,
                'vector': query_list,
                'limit': top_k,
                'with_payload': True,
                'with_vectors': False,
                'query_filter': query_filter
            }

            for method_name in candidate_names:
                if hasattr(self.client, method_name):
                    attempted_methods.append(method_name)
                    method = getattr(self.client, method_name)
                    try:
                        sig = inspect.signature(method)
                        # choose kwargs supported by this method
                        kwargs = {k: v for k, v in possible_kwargs.items() if k in sig.parameters and v is not None}
                        # If method signature does not accept a query vector keyword, prefer positional call
                        accepts_vector_kw = any(k in kwargs for k in ('query_vector', 'vector'))
                        if not accepts_vector_kw:
                            # Try positional fallback first: common positional ordering (collection_name, query_vector, limit)
                            try:
                                results = method(self.collection_name, query_list, top_k)
                                logger.info(f"Vector search using client.{method_name}(collection, vector, limit) [positional]")
                                break
                            except Exception as e:
                                logger.debug(f"client.{method_name} positional call failed: {e}")
                        # Try keyword call (if vector kw present or positional failed)
                        try:
                            results = method(**kwargs)
                            logger.info(f"Vector search using client.{method_name}() with kwargs {list(kwargs.keys())}")
                            break
                        except TypeError:
                            # positional fallback if keyword call fails
                            results = method(self.collection_name, query_list, top_k)
                            logger.info(f"Vector search using client.{method_name}(collection, vector, limit) [positional after TypeError]")
                            break
                    except TypeError:
                        # positional fallback: try common positional ordering (collection_name, query_vector, limit)
                        try:
                            results = method(self.collection_name, query_list, top_k)
                            logger.info(f"Vector search using client.{method_name}(collection, vector, limit) [positional in except]")
                            break
                        except Exception as e:
                            logger.debug(f"client.{method_name} positional call failed: {e}")
                    except Exception as e:
                        logger.debug(f"client.{method_name} call failed: {e}")

            if results is None:
                logger.error(f"No compatible search method found on Qdrant client. Attempted: {attempted_methods}")
                raise AttributeError("No compatible search method found on Qdrant client")
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        # Format results
        formatted_results = []
        try:
            for result in results:
                # Log short repr and available attributes for debugging
                try:
                    rrepr = repr(result)
                except Exception:
                    rrepr = f'<unrepresentable {type(result)}>'
                attrs = [a for a in dir(result) if not a.startswith('_')]
                logger.debug(f"Result repr={rrepr[:300]}, attrs={attrs}")

                # Attempt to extract id/score/payload robustly from multiple shapes
                rid = None
                score = None
                payload = None

                # tuple-like (id, score, payload)
                if isinstance(result, tuple) and len(result) >= 3:
                    rid = result[0]
                    score = result[1]
                    payload = result[2]

                # object with attributes
                if rid is None and hasattr(result, 'id'):
                    try:
                        rid = getattr(result, 'id')
                    except Exception:
                        rid = None
                if score is None and hasattr(result, 'score'):
                    try:
                        score = getattr(result, 'score')
                    except Exception:
                        score = None
                if payload is None and hasattr(result, 'payload'):
                    try:
                        payload = getattr(result, 'payload')
                    except Exception:
                        payload = None

                # alternate attribute name
                if rid is None and hasattr(result, 'point_id'):
                    try:
                        rid = getattr(result, 'point_id')
                    except Exception:
                        pass

                # mapping-like
                if rid is None and hasattr(result, 'get'):
                    try:
                        rid = result.get('id', result.get('point_id', None))
                    except Exception:
                        pass

                # fallback: empty payload
                payload = payload or {}
                logger.debug(f"Extracted rid={rid}, score={score}, payload_type={type(payload)}")
                if ('ciphertext' not in (payload or {})) and rid is None:
                    # Log at INFO so it's visible by default
                    try:
                        logger.info(f"Result with empty payload and no id. repr: {repr(result)[:500]}")
                    except Exception:
                        logger.info(f"Result with empty payload and no id of type {type(result)}")

                # payload may be dict-like or have .to_dict()
                try:
                    if hasattr(payload, 'to_dict'):
                        payload_dict = payload.to_dict()
                    else:
                        payload_dict = dict(payload) if payload is not None else {}
                except Exception:
                    payload_dict = {}
                # If ciphertext/nonce missing, log payload content at INFO to help debugging
                keys = list(payload_dict.keys())
                if 'ciphertext' not in payload_dict or 'nonce' not in payload_dict:
                    logger.info(f"Search result payload keys (missing ciphertext/nonce): {keys}")
                    # show some values for likely text fields
                    preview = {k: (str(v)[:200] + '...') if isinstance(v, (str, bytes)) and len(str(v))>200 else v for k, v in list(payload_dict.items())[:10]}
                    logger.info(f"Payload preview: {preview}")
                else:
                    logger.debug(f"Parsed payload keys: {keys}")

                formatted_results.append({
                    'id': rid,
                    'score': score,
                    'ciphertext': payload_dict.get('ciphertext'),
                    'nonce': payload_dict.get('nonce'),
                    'metadata': {k: v for k, v in payload_dict.items() if k not in ['ciphertext', 'nonce']}
                })
        except Exception as e:
            logger.error(f"Failed to format search results: {e}")
            return []

        # If payloads are empty (no ciphertext/nonce found), try to fetch payloads via retrieve() for the returned ids
        try:
            has_cipher = any(item.get('ciphertext') for item in formatted_results)
            if not has_cipher:
                ids = [item.get('id') for item in formatted_results if item.get('id') is not None]
                if ids:
                    try:
                        # retrieve detailed records by ids
                        records = self.client.retrieve(collection_name=self.collection_name, ids=ids, with_payload=True, with_vectors=False)
                        # reformat retrieved records
                        formatted_results = []
                        for rec in records:
                            payload = getattr(rec, 'payload', None) or (rec.payload if hasattr(rec, 'payload') else {})
                            try:
                                if hasattr(payload, 'to_dict'):
                                    pd = payload.to_dict()
                                else:
                                    pd = dict(payload) if payload is not None else {}
                            except Exception:
                                pd = {}

                            formatted_results.append({
                                'id': getattr(rec, 'id', None),
                                'score': getattr(rec, 'score', None),
                                'ciphertext': pd.get('ciphertext'),
                                'nonce': pd.get('nonce'),
                                'metadata': {k: v for k, v in pd.items() if k not in ['ciphertext', 'nonce']}
                            })
                        logger.info('Fetched payloads via client.retrieve and reformatted results')
                    except Exception as e:
                        logger.debug(f'client.retrieve fallback failed: {e}')
        except Exception:
            pass

        # If still no ciphertexts, perform a local scan: fetch points with vectors and payloads and compute similarity locally
        try:
            has_cipher = any(item.get('ciphertext') for item in formatted_results)
            if not has_cipher:
                logger.info('Attempting local scan fallback: retrieving vectors and payloads via scroll')
                try:
                    records = self.client.scroll(collection_name=self.collection_name, limit=10000, with_payload=True, with_vectors=True)
                    # records may be tuple (list, next)
                    recs = records[0] if isinstance(records, tuple) else records
                except Exception as e:
                    logger.debug(f'client.scroll failed: {e}')
                    recs = []

                # Collect vectors and payloads
                vecs = []
                rec_meta = []
                for rec in recs:
                    try:
                        vec = getattr(rec, 'vector', None)
                        if vec is None and hasattr(rec, 'get'):
                            vec = rec.get('vector')
                        if vec is None:
                            continue
                        # convert to numpy
                        v = np.asarray(vec)
                        payload = getattr(rec, 'payload', None) or (rec.payload if hasattr(rec, 'payload') else {})
                        # normalize payload dict
                        try:
                            if hasattr(payload, 'to_dict'):
                                pd = payload.to_dict()
                            else:
                                pd = dict(payload) if payload is not None else {}
                        except Exception:
                            pd = {}

                        vecs.append(v)
                        rec_meta.append((getattr(rec, 'id', None), pd, getattr(rec, 'score', None)))
                    except Exception:
                        continue

                if vecs:
                    mat = np.vstack(vecs)
                    qv = np.asarray(query_list)
                    # normalize
                    try:
                        qv_norm = qv / np.linalg.norm(qv)
                        mat_norms = mat / np.linalg.norm(mat, axis=1, keepdims=True)
                        sims = mat_norms.dot(qv_norm)
                    except Exception:
                        sims = np.dot(mat, qv)

                    # pick top_k
                    idxs = np.argsort(-sims)[:top_k]
                    formatted_results = []
                    for ix in idxs:
                        pid, pd, sc = rec_meta[ix]
                        formatted_results.append({
                            'id': pid,
                            'score': float(sims[ix]) if sims is not None else sc,
                            'ciphertext': pd.get('ciphertext'),
                            'nonce': pd.get('nonce'),
                            'metadata': {k: v for k, v in pd.items() if k not in ['ciphertext', 'nonce']}
                        })
                    logger.info(f'Local scan fallback selected {len(formatted_results)} results')
        except Exception as e:
            logger.debug(f'Local scan fallback failed: {e}')

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
            'vectors_count': info.points_count,
            'points_count': info.points_count,
            'status': info.status
        }

    def count(self) -> int:
        """Get number of vectors in collection"""
        info = self.client.get_collection(collection_name=self.collection_name)
        return info.points_count
