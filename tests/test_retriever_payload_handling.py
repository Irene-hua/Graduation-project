import unittest

from src.retrieval.retriever import Retriever


class FakeEncryption:
    def decrypt(self, ciphertext, nonce):
        if ciphertext == 'valid_ct' and nonce == 'valid_nonce':
            return 'decrypted text'
        raise ValueError('decrypt failed')


class FakeVectorStore:
    def _normalize_payload(self, payload):
        return payload


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


if __name__ == '__main__':
    unittest.main()

