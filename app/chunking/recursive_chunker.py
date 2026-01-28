from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.chunking.base_chunker import BaseChunker, Chunk
from app.loaders.base_loader import Document


class RecursiveChunker(BaseChunker):
    """Recursive character-based chunker."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk document using recursive splitting."""
        texts = self.splitter.split_text(document.content)
        
        chunks = []
        for idx, text in enumerate(texts):
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "chunking_strategy": "recursive",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            })
            
            chunks.append(Chunk(
                content=text,
                metadata=chunk_metadata,
                chunk_index=idx
            ))
        
        return chunks
    
    def get_name(self) -> str:
        return "recursive"
