from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from app.chunking.base_chunker import Chunk


class SearchResult:
    """Represents a search result."""
    def __init__(
        self,
        chunk: Chunk,
        score: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.chunk = chunk
        self.score = score
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.3f}, chunk_index={self.chunk.chunk_index})"


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], embedder) -> None:
        """Add chunks to the vector store."""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int, embedder) -> List[SearchResult]:
        """Search for similar chunks."""
        pass
    
    @abstractmethod
    def delete_all(self) -> None:
        """Delete all vectors from the store."""
        pass
    
    @abstractmethod
    def get_count(self) -> int:
        """Get the number of vectors in the store."""
        pass
