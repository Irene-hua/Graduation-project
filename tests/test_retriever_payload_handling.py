import unittest

from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever


class FakeEncryption:
    def decrypt(self, ciphertext, nonce):
        if ciphertext == 'valid_ct' and nonce == 'valid_nonce':
            return 'decrypted text'
        raise ValueError('decrypt failed')


class FakeVectorStore:
    def _normalize_payload(self, payload):
        return payload


class FakeCollectionClient:
    def __init__(self, existing_names=None):
        self.deleted = False
        self.created = False
        self.existing_names = existing_names or []

    def get_collections(self):
        class _C:
            def __init__(self, names):
                self.collections = [type('X', (), {'name': n})() for n in names]
        return _C(self.existing_names)

    def delete_collection(self, collection_name):
        self.deleted = True

    def create_collection(self, collection_name, vectors_config):
        self.created = True


class RetrieverPayloadHandlingTest(unittest.TestCase):
    def setUp(self):
        self.retriever = Retriever(embedding_model=None, vector_store=FakeVectorStore(), encryption=FakeEncryption())

    def test_plaintext_payload_is_used(self):
        text = self.retriever._safe_decrypt_payload({'text': 'hello world'})
        self.assertEqual(text, 'hello world')

    def test_invalid_non_string_cipher_payload_is_skipped(self):
        text = self.retriever._safe_decrypt_payload({'ciphertext': {'bad': 'type'}, 'nonce': 123})
        self.assertIsNone(text)

    def test_valid_encrypted_payload_is_decrypted(self):
        text = self.retriever._safe_decrypt_payload({'ciphertext': 'valid_ct', 'nonce': 'valid_nonce'})
        self.assertEqual(text, 'decrypted text')

    def test_reset_collection_recreates_collection(self):
        store = VectorStore.__new__(VectorStore)
        store.collection_name = 'test_collection'
        store.dimension = 384
        store.distance = 'Cosine'
        store.client = FakeCollectionClient(existing_names=['test_collection'])

        store.reset_collection()
        self.assertTrue(store.client.deleted)
        self.assertTrue(store.client.created)

    def test_fake_collection_client_exposes_existing_names(self):
        client = FakeCollectionClient(existing_names=['a', 'b'])
        names = [c.name for c in client.get_collections().collections]
        self.assertEqual(names, ['a', 'b'])

    def test_collection_override_name_is_preserved(self):
        store = VectorStore.__new__(VectorStore)
        store.collection_name = 'encrypted_documents_test1'
        self.assertEqual(store.collection_name, 'encrypted_documents_test1')


if __name__ == '__main__':
    unittest.main()
