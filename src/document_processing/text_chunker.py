"""
Text Chunker
Splits text into chunks with configurable size and overlap
"""

from typing import List, Dict
import re


class TextChunker:
    """Split text into manageable chunks for embedding and storage"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with 'text', 'chunk_id', and metadata
        """
        if not text or not text.strip():
            return []
        
        # Clean text
        text = self._clean_text(text)
        
        chunks = []
        start_idx = 0
        chunk_id = 0
        
        while start_idx < len(text):
            # Calculate end index for this chunk
            end_idx = start_idx + self.chunk_size
            
            # If we're not at the end, try to break at sentence or word boundary
            if end_idx < len(text):
                end_idx = self._find_break_point(text, start_idx, end_idx)
            
            # Extract chunk
            chunk_text = text[start_idx:end_idx].strip()
            
            if chunk_text:
                chunk_dict = {
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'length': len(chunk_text)
                }
                
                # Add metadata if provided
                if metadata:
                    chunk_dict.update(metadata)
                
                chunks.append(chunk_dict)
                chunk_id += 1
            
            # Move to next chunk with overlap
            start_idx = end_idx - self.chunk_overlap
            
            # Ensure we make progress
            if start_idx <= chunks[-1]['start_idx'] if chunks else 0:
                start_idx = end_idx
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def _find_break_point(self, text: str, start: int, preferred_end: int) -> int:
        """
        Find a good break point (sentence or word boundary) near preferred_end
        
        Args:
            text: Full text
            start: Start index of current chunk
            preferred_end: Preferred end index
            
        Returns:
            Adjusted end index at a natural break point
        """
        # Look for sentence boundaries first (within 100 chars of preferred end)
        search_start = max(start, preferred_end - 100)
        search_text = text[search_start:preferred_end + 50]
        
        # Look for sentence ending punctuation
        sentence_endings = ['. ', '! ', '? ', '。', '！', '？', '\n\n']
        
        best_break = -1
        for ending in sentence_endings:
            pos = search_text.rfind(ending)
            if pos != -1:
                best_break = max(best_break, search_start + pos + len(ending))
        
        if best_break > start:
            return best_break
        
        # If no sentence boundary, look for word boundary
        search_text = text[preferred_end - 50:preferred_end + 50]
        word_breaks = [' ', '\n', '\t']
        
        for i in range(len(search_text) // 2, 0, -1):
            if search_text[i] in word_breaks:
                return preferred_end - 50 + i + 1
        
        # Fall back to preferred end
        return preferred_end
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of document dicts with 'content' key
            
        Returns:
            List of all chunks with document metadata
        """
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            content = doc.get('content', '')
            
            # Prepare metadata
            metadata = {
                'doc_id': doc_idx,
                'source_file': doc.get('filename', doc.get('filepath', f'doc_{doc_idx}'))
            }
            
            # Add any other metadata from document
            for key, value in doc.items():
                if key not in ['content', 'doc_id']:
                    metadata[f'doc_{key}'] = value
            
            # Chunk the document
            doc_chunks = self.chunk_text(content, metadata)
            
            # Add global chunk ID
            for chunk in doc_chunks:
                chunk['global_chunk_id'] = len(all_chunks)
                all_chunks.append(chunk)
        
        return all_chunks
