"""
AES Encryption/Decryption Module for Text Chunks
Provides secure encryption for document chunks before storage
"""

import os
import base64
from typing import Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import hashlib


class AESEncryption:
    """AES-GCM encryption for text data"""
    
    def __init__(self, key_size: int = 256):
        """
        Initialize AES encryption
        
        Args:
            key_size: Key size in bits (128, 192, or 256)
        """
        self.key_size = key_size
        self.key_bytes = key_size // 8
        self.key = None
        
    def generate_key(self) -> bytes:
        """Generate a new random encryption key"""
        self.key = AESGCM.generate_key(bit_length=self.key_size)
        return self.key
    
    def set_key(self, key: bytes):
        """Set encryption key from existing key"""
        if len(key) != self.key_bytes:
            raise ValueError(f"Key must be {self.key_bytes} bytes")
        self.key = key
    
    def derive_key_from_password(self, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """
        Derive encryption key from password using PBKDF2
        
        Args:
            password: User password
            salt: Salt for key derivation (generated if None)
            
        Returns:
            Tuple of (key, salt)
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=self.key_bytes,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(password.encode())
        self.key = key
        return key, salt
    
    def encrypt(self, plaintext: str) -> Tuple[str, str]:
        """
        Encrypt plaintext using AES-GCM
        
        Args:
            plaintext: Text to encrypt
            
        Returns:
            Tuple of (ciphertext_base64, nonce_base64)
        """
        if self.key is None:
            raise ValueError("Encryption key not set. Call generate_key() or set_key() first")
        
        # Convert text to bytes
        plaintext_bytes = plaintext.encode('utf-8')
        
        # Generate nonce (12 bytes for GCM)
        nonce = os.urandom(12)
        
        # Encrypt
        aesgcm = AESGCM(self.key)
        ciphertext = aesgcm.encrypt(nonce, plaintext_bytes, None)
        
        # Encode to base64 for storage
        ciphertext_b64 = base64.b64encode(ciphertext).decode('utf-8')
        nonce_b64 = base64.b64encode(nonce).decode('utf-8')
        
        return ciphertext_b64, nonce_b64
    
    def decrypt(self, ciphertext_b64: str, nonce_b64: str) -> str:
        """
        Decrypt ciphertext using AES-GCM
        
        Args:
            ciphertext_b64: Base64-encoded ciphertext
            nonce_b64: Base64-encoded nonce
            
        Returns:
            Decrypted plaintext
        """
        if self.key is None:
            raise ValueError("Encryption key not set")
        
        # Decode from base64
        ciphertext = base64.b64decode(ciphertext_b64)
        nonce = base64.b64decode(nonce_b64)
        
        # Decrypt
        aesgcm = AESGCM(self.key)
        plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, None)
        
        # Convert bytes to text
        plaintext = plaintext_bytes.decode('utf-8')
        
        return plaintext
    
    def save_key(self, filepath: str):
        """Save encryption key to file"""
        if self.key is None:
            raise ValueError("No key to save")
        
        with open(filepath, 'wb') as f:
            f.write(self.key)
    
    def load_key(self, filepath: str):
        """Load encryption key from file"""
        with open(filepath, 'rb') as f:
            self.key = f.read()
        
        if len(self.key) != self.key_bytes:
            raise ValueError(f"Invalid key size: expected {self.key_bytes} bytes")
    
    def get_key_hash(self) -> str:
        """Get SHA256 hash of current key for verification"""
        if self.key is None:
            raise ValueError("No key set")
        return hashlib.sha256(self.key).hexdigest()
