from abc import ABC, abstractmethod
from typing import List, Dict, Any

from app.loaders.base_loader import Document


class Chunk:
    """Represents a text chunk."""
    def __init__(
        self,
        content: str,
        metadata: Dict[str, Any],
        chunk_index: int = 0
    ):
        self.content = content
        self.metadata = metadata
        self.chunk_index = chunk_index
    
    def __repr__(self) -> str:
        return f"Chunk(index={self.chunk_index}, length={len(self.content)})"


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk a document into smaller pieces."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this chunking strategy."""
        pass
