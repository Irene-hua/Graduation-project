"""
Unit tests for document processing module
"""

import unittest
from src.document_processing import TextChunker


class TestTextChunker(unittest.TestCase):
    """Test cases for text chunking"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        self.sample_text = """
        Machine learning is a subset of artificial intelligence. 
        It focuses on developing algorithms that can learn from data.
        Deep learning uses neural networks with multiple layers.
        Natural language processing deals with text and speech.
        """
    
    def test_basic_chunking(self):
        """Test basic text chunking"""
        chunks = self.chunker.chunk_text(self.sample_text)
        
        # Should produce multiple chunks
        self.assertGreater(len(chunks), 0)
        
        # Each chunk should have required fields
        for chunk in chunks:
            self.assertIn('chunk_id', chunk)
            self.assertIn('text', chunk)
            self.assertIn('start_idx', chunk)
            self.assertIn('end_idx', chunk)
    
    def test_chunk_size_limit(self):
        """Test that chunks respect size limit"""
        chunks = self.chunker.chunk_text(self.sample_text)
        
        for chunk in chunks:
            # Allow some tolerance for boundary finding
            self.assertLessEqual(len(chunk['text']), self.chunker.chunk_size + 50)
    
    def test_empty_text(self):
        """Test chunking empty text"""
        chunks = self.chunker.chunk_text("")
        self.assertEqual(len(chunks), 0)
        
        chunks = self.chunker.chunk_text("   ")
        self.assertEqual(len(chunks), 0)
    
    def test_short_text(self):
        """Test chunking text shorter than chunk size"""
        short_text = "This is a short text."
        chunks = self.chunker.chunk_text(short_text)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['text'], short_text)
    
    def test_chunk_documents(self):
        """Test chunking multiple documents"""
        documents = [
            {'content': 'Document 1 content', 'filename': 'doc1.txt'},
            {'content': 'Document 2 content', 'filename': 'doc2.txt'}
        ]
        
        chunks = self.chunker.chunk_documents(documents)
        
        self.assertGreater(len(chunks), 0)
        
        # Check metadata is preserved
        for chunk in chunks:
            self.assertIn('source_file', chunk)
            self.assertIn('doc_id', chunk)
            self.assertIn('global_chunk_id', chunk)


if __name__ == '__main__':
    unittest.main()
