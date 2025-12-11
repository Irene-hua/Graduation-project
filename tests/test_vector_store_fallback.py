import unittest
from unittest.mock import MagicMock
import numpy as np

from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.encryption.aes_encryption import AESEncryption


class DummyEmbedding:
    def __init__(self, dim=384):
        self._dim = dim
    def encode(self, text, batch_size=1, show_progress=False):
        # return deterministic vector
        return np.ones((1, self._dim), dtype=float)
    def get_dimension(self):
        return self._dim


class DummyRecord:
    def __init__(self, id, vector, payload, score=0.1):
        self.id = id
        self.vector = vector
        self.payload = payload
        self.score = score


class TestVectorStoreFallback(unittest.TestCase):
    def test_local_scroll_fallback_and_decrypt(self):
        # Setup encryption and key
        enc = AESEncryption(key_size=256)
        key = enc.generate_key()
        enc.set_key(key)

        # Prepare a plaintext and encrypt it
        plaintext = 'This is a test chunk. Planned improvements include: - A - B - C'
        ciphertext_b64, nonce_b64 = enc.encrypt(plaintext)

        # Create dummy records as qdrant would return
        payload = {'ciphertext': ciphertext_b64, 'nonce': nonce_b64, 'chunk_id': 1, 'source_file': 'test.txt'}
        vec = np.ones(384).tolist()
        records = [DummyRecord('id1', vec, payload, score=0.5)]

        # Mock client with scroll
        mock_client = MagicMock()
        mock_client.scroll.return_value = (records, None)
        mock_client.get_collections.return_value = MagicMock(collections=[])

        # Create VectorStore and inject mock client
        vs = VectorStore(collection_name='test', dimension=384, storage_path='.')
        vs.client = mock_client

        # Run search which should use local scroll fallback
        qv = np.ones(384)
        results = vs.search(qv, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['ciphertext'], ciphertext_b64)
        self.assertEqual(results[0]['nonce'], nonce_b64)

        # Test Retriever decrypts correctly
        dummy_embed = DummyEmbedding()
        retriever = Retriever(dummy_embed, vs, enc)
        dec = retriever.retrieve('test query', top_k=1)
        self.assertEqual(len(dec), 1)
        self.assertIn('Planned improvements include', dec[0]['text'])


if __name__ == '__main__':
    unittest.main()
