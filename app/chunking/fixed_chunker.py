from typing import List

from app.chunking.base_chunker import BaseChunker, Chunk
from app.loaders.base_loader import Document


class FixedChunker(BaseChunker):
    """Fixed-size character chunker."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk document into fixed-size pieces."""
        content = document.content
        chunks = []
        start = 0
        idx = 0
        
        while start < len(content):
            end = start + self.chunk_size
            chunk_content = content[start:end]
            
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "chunking_strategy": "fixed",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "start_position": start,
                "end_position": end,
            })
            
            chunks.append(Chunk(
                content=chunk_content,
                metadata=chunk_metadata,
                chunk_index=idx
            ))
            
            start = end - self.chunk_overlap
            idx += 1
        
        return chunks
    
    def get_name(self) -> str:
        return "fixed"
