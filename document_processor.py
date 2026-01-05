"""Document processing utilities for handling text data."""
import re
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""
    text: str
    metadata: dict = None
    chunk_id: str = ""

class DocumentProcessor:
    """Handles document processing tasks like chunking and cleaning."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize with chunking parameters."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def chunk_document(self, text: str, metadata: Optional[dict] = None) -> List[DocumentChunk]:
        """Split document into overlapping chunks."""
        if not text:
            return []
            
        text = self.clean_text(text)
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            # Create chunk with metadata
            chunk_meta = metadata.copy() if metadata else {}
            chunk_meta.update({
                'chunk_id': chunk_id,
                'start_pos': start,
                'end_pos': end
            })
            
            chunks.append(DocumentChunk(
                text=chunk_text,
                metadata=chunk_meta,
                chunk_id=str(chunk_id)
            ))
            
            # Move the window
            if end == len(text):
                break
                
            start = end - self.chunk_overlap
            chunk_id += 1
            
        return chunks
