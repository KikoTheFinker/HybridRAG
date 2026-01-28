from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """Abstract base class for embedding services."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings."""
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """Get the dimension of the embeddings."""
        pass
