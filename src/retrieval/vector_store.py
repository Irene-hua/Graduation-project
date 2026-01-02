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
import json

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

    # -------------------- Payload normalization helpers --------------------
    def _extract_value(self, value):
        """
        Extract a primitive Python value from various qdrant/protobuf wrapper objects.
        """
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool, bytes)):
            if isinstance(value, bytes):
                try:
                    return value.decode('utf-8')
                except Exception:
                    return value
            return value

        if hasattr(value, 'to_dict'):
            try:
                return value.to_dict()
            except Exception:
                pass
        if hasattr(value, 'to_json'):
            try:
                return json.loads(value.to_json())
            except Exception:
                pass

        # Common protobuf Value fields
        for attr in ('string_value', 'text_value', 'integer_value', 'int_value', 'bool_value', 'float_value', 'double_value', 'bytes_value'):
            if hasattr(value, attr):
                v = getattr(value, attr)
                if v is not None:
                    if isinstance(v, bytes):
                        try:
                            return v.decode('utf-8')
                        except Exception:
                            return v
                    return v

        if hasattr(value, 'value'):
            try:
                return self._extract_value(getattr(value, 'value'))
            except Exception:
                pass

        try:
            return str(value)
        except Exception:
            return None

    def _normalize_payload(self, payload) -> Dict:
        """
        Convert various payload shapes returned by qdrant-client into a plain dict
        of primitive Python values.
        """
        if payload is None:
            return {}

        if isinstance(payload, dict):
            out = {}
            for k, v in payload.items():
                out[k] = self._extract_value(v)
            return out

        if hasattr(payload, 'to_dict'):
            try:
                d = payload.to_dict()
                if isinstance(d, dict):
                    return {k: self._extract_value(v) for k, v in d.items()}
            except Exception:
                pass

        if hasattr(payload, 'items'):
            try:
                return {k: self._extract_value(v) for k, v in payload.items()}
            except Exception:
                pass

        if isinstance(payload, (list, tuple)):
            out = {}
            for entry in payload:
                key = None
                val = None
                if hasattr(entry, 'key') and hasattr(entry, 'value'):
                    key = getattr(entry, 'key')
                    val = getattr(entry, 'value')
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    key, val = entry[0], entry[1]
                else:
                    key = getattr(entry, 'name', None) or getattr(entry, 'k', None)
                    val = getattr(entry, 'v', None) or getattr(entry, 'value', None)

                if key is None:
                    continue
                out[key] = self._extract_value(val)
            return out

        if hasattr(payload, 'payload'):
            try:
                return self._normalize_payload(getattr(payload, 'payload'))
            except Exception:
                pass

        try:
            d = dict(payload)
            return {k: self._extract_value(v) for k, v in d.items()}
        except Exception:
            return {}

    # -------------------- Search --------------------
    def search(self,
               query_vector: np.ndarray,
               top_k: int = 5,
               filter_dict: Dict = None) -> List[Dict]:
        """
        Search for similar vectors and return formatted results with payloads
        """
        # Normalize and validate query_vector
        try:
            if isinstance(query_vector, list):
                query_vec = np.array(query_vector)
            else:
                query_vec = np.asarray(query_vector)

            if query_vec.ndim == 2 and query_vec.shape[0] == 1:
                query_vec = query_vec[0]

            if query_vec.ndim != 1:
                raise ValueError("query_vector must be 1D or shape (1, dim)")

            if query_vec.shape[0] != self.dimension:
                raise ValueError("Query vector dimension does not match store dimension")

            query_list = query_vec.tolist()
        except Exception as e:
            logger.error(f"Invalid query vector provided: {e}")
            return []

        query_filter = None
        if filter_dict:
            logger.debug("Filters supplied to VectorStore.search but filtering is not implemented")
            raise NotImplementedError("Filtering is not yet implemented.")

        try:
            # Prefer local scroll + local similarity for local storage to guarantee payloads available
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
                        pd = self._normalize_payload(payload)
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

            # Try several possible qdrant client methods to support multiple versions
            results = None
            attempted_methods = []

            candidate_names = [
                'search', 'search_points', 'search_collection', 'search_batch', 'search_by_vector',
                'query', 'query_points', 'query_batch', 'query_points_groups',
                'retrieve'
            ]

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
                        kwargs = {k: v for k, v in possible_kwargs.items() if k in sig.parameters and v is not None}
                        accepts_vector_kw = any(k in kwargs for k in ('query_vector', 'vector'))
                        if not accepts_vector_kw:
                            try:
                                results = method(self.collection_name, query_list, top_k)
                                logger.info(f"Vector search using client.{method_name}(collection, vector, limit) [positional]")
                                break
                            except Exception as e:
                                logger.debug(f"client.{method_name} positional call failed: {e}")
                        try:
                            results = method(**kwargs)
                            logger.info(f"Vector search using client.{method_name}() with kwargs {list(kwargs.keys())}")
                            break
                        except TypeError:
                            results = method(self.collection_name, query_list, top_k)
                            logger.info(f"Vector search using client.{method_name}(collection, vector, limit) [positional after TypeError]")
                            break
                    except TypeError:
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

        # Format results from whatever shape 'results' took
        formatted_results = []
        try:
            # Normalize wrapper shapes
            if hasattr(results, 'result') and isinstance(getattr(results, 'result'), (list, tuple)):
                results_iter = getattr(results, 'result')
            elif isinstance(results, dict) and 'result' in results and isinstance(results['result'], (list, tuple)):
                results_iter = results['result']
            elif isinstance(results, (list, tuple)) and len(results) >= 2 and isinstance(results[1], (list, tuple)):
                # e.g. ('points', [ScoredPoint,...])
                results_iter = results[1]
            else:
                results_iter = results

            for result in results_iter:
                try:
                    # Try to extract id/score/payload from many shapes
                    rid = None
                    score = None
                    payload = None

                    if isinstance(result, tuple) and len(result) >= 3:
                        rid = result[0]
                        score = result[1]
                        payload = result[2]

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

                    if rid is None and hasattr(result, 'point_id'):
                        try:
                            rid = getattr(result, 'point_id')
                        except Exception:
                            pass

                    if rid is None and hasattr(result, 'get'):
                        try:
                            rid = result.get('id', result.get('point_id', None))
                        except Exception:
                            pass

                    payload = payload or {}
                    logger.debug(f"Extracted rid={rid}, score={score}, payload_type={type(payload)}")

                    # Normalize payload into plain dict
                    try:
                        payload_dict = self._normalize_payload(payload)
                    except Exception:
                        payload_dict = {}

                    keys = list(payload_dict.keys())
                    if 'ciphertext' not in payload_dict or 'nonce' not in payload_dict:
                        logger.info(f"Search result payload keys (missing ciphertext/nonce): {keys}")
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
                    logger.debug(f"Failed to parse individual search result: {e}")
                    continue
        except Exception as e:
            logger.error(f"Failed to format search results: {e}")
            return []

        # If no ciphertexts, try retrieve(ids) to fetch payloads
        try:
            has_cipher = any(item.get('ciphertext') for item in formatted_results)
            if not has_cipher:
                ids = [item.get('id') for item in formatted_results if item.get('id') is not None]
                if ids:
                    try:
                        records = self.client.retrieve(collection_name=self.collection_name, ids=ids, with_payload=True, with_vectors=False)
                        formatted_results = []
                        for rec in records:
                            payload = getattr(rec, 'payload', None) or (rec.payload if hasattr(rec, 'payload') else {})
                            pd = self._normalize_payload(payload)
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

        # If still no ciphertexts, do a local scan fallback computing similarity locally
        try:
            has_cipher = any(item.get('ciphertext') for item in formatted_results)
            if not has_cipher:
                logger.info('Attempting local scan fallback: retrieving vectors and payloads via scroll')
                try:
                    records = self.client.scroll(collection_name=self.collection_name, limit=10000, with_payload=True, with_vectors=True)
                    recs = records[0] if isinstance(records, tuple) else records
                except Exception as e:
                    logger.debug(f'client.scroll failed: {e}')
                    recs = []

                vecs = []
                rec_meta = []
                for rec in recs:
                    try:
                        vec = getattr(rec, 'vector', None)
                        if vec is None and hasattr(rec, 'get'):
                            vec = rec.get('vector')
                        if vec is None:
                            continue
                        v = np.asarray(vec)
                        payload = getattr(rec, 'payload', None) or (rec.payload if hasattr(rec, 'payload') else {})
                        pd = self._normalize_payload(payload)
                        vecs.append(v)
                        rec_meta.append((getattr(rec, 'id', None), pd, getattr(rec, 'score', None)))
                    except Exception:
                        continue

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
