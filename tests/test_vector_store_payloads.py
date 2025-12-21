import unittest
import numpy as np
from src.retrieval.vector_store import VectorStore


class DummyEntry:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class DummyScoredPoint:
    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload


class TestVectorStorePayloads(unittest.TestCase):
    def test_normalize_payload_dict(self):
        vs = VectorStore()
        payload = {'ciphertext': 'abc', 'nonce': '123', 'chunk_id': 1}
        out = vs._normalize_payload(payload)
        self.assertEqual(out['ciphertext'], 'abc')
        self.assertEqual(out['nonce'], '123')
        self.assertEqual(out['chunk_id'], 1)

    def test_normalize_payload_list_entries(self):
        vs = VectorStore()
        entries = [DummyEntry('ciphertext', 'xyz'), DummyEntry('nonce', '999')]
        out = vs._normalize_payload(entries)
        self.assertEqual(out['ciphertext'], 'xyz')
        self.assertEqual(out['nonce'], '999')

    def test_local_search_returns_payloads(self):
        # This test will attempt to run a local search against the existing qdrant storage
        vs = VectorStore()
        # Create a dummy random query vector of correct dimension
        qv = np.random.randn(vs.dimension).astype(float)
        results = vs.search(qv, top_k=3)
        self.assertIsInstance(results, list)
        for r in results:
            self.assertIn('id', r)
            self.assertIn('ciphertext', r)
            self.assertIn('nonce', r)


if __name__ == '__main__':
    unittest.main()
