"""
Unit tests for encryption module
"""

import unittest
import os
import tempfile
from src.encryption import AESEncryption


class TestAESEncryption(unittest.TestCase):
    """Test cases for AES encryption"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.encryption = AESEncryption(key_size=256)
        self.encryption.generate_key()
        self.test_text = "This is a test message for encryption."
    
    def test_encrypt_decrypt(self):
        """Test basic encryption and decryption"""
        # Encrypt
        ciphertext, nonce = self.encryption.encrypt(self.test_text)
        
        # Verify ciphertext is different from plaintext
        self.assertNotEqual(ciphertext, self.test_text)
        
        # Decrypt
        decrypted = self.encryption.decrypt(ciphertext, nonce)
        
        # Verify decrypted matches original
        self.assertEqual(decrypted, self.test_text)
    
    def test_different_ciphertexts(self):
        """Test that same plaintext produces different ciphertexts"""
        ciphertext1, nonce1 = self.encryption.encrypt(self.test_text)
        ciphertext2, nonce2 = self.encryption.encrypt(self.test_text)
        
        # Different nonces should produce different ciphertexts
        self.assertNotEqual(ciphertext1, ciphertext2)
        self.assertNotEqual(nonce1, nonce2)
        
        # Both should decrypt to original
        decrypted1 = self.encryption.decrypt(ciphertext1, nonce1)
        decrypted2 = self.encryption.decrypt(ciphertext2, nonce2)
        self.assertEqual(decrypted1, self.test_text)
        self.assertEqual(decrypted2, self.test_text)
    
    def test_key_save_load(self):
        """Test saving and loading encryption key"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            key_file = f.name
        
        try:
            # Save key
            self.encryption.save_key(key_file)
            
            # Create new encryption instance and load key
            new_encryption = AESEncryption(key_size=256)
            new_encryption.load_key(key_file)
            
            # Encrypt with original, decrypt with new
            ciphertext, nonce = self.encryption.encrypt(self.test_text)
            decrypted = new_encryption.decrypt(ciphertext, nonce)
            
            self.assertEqual(decrypted, self.test_text)
        finally:
            if os.path.exists(key_file):
                os.remove(key_file)
    
    def test_key_derivation(self):
        """Test key derivation from password"""
        password = "test_password_123"
        
        key1, salt = self.encryption.derive_key_from_password(password)
        
        # Create new instance with same password and salt
        new_encryption = AESEncryption(key_size=256)
        key2, _ = new_encryption.derive_key_from_password(password, salt)
        
        # Keys should be identical
        self.assertEqual(key1, key2)
        
        # Should be able to decrypt
        ciphertext, nonce = self.encryption.encrypt(self.test_text)
        decrypted = new_encryption.decrypt(ciphertext, nonce)
        self.assertEqual(decrypted, self.test_text)
    
    def test_unicode_text(self):
        """Test encryption of unicode text"""
        unicode_text = "测试中文加密功能 🔒"
        
        ciphertext, nonce = self.encryption.encrypt(unicode_text)
        decrypted = self.encryption.decrypt(ciphertext, nonce)
        
        self.assertEqual(decrypted, unicode_text)
    
    def test_long_text(self):
        """Test encryption of long text"""
        long_text = "A" * 10000
        
        ciphertext, nonce = self.encryption.encrypt(long_text)
        decrypted = self.encryption.decrypt(ciphertext, nonce)
        
        self.assertEqual(decrypted, long_text)


if __name__ == '__main__':
    unittest.main()
