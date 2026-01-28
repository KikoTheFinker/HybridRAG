from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any


class Document:
    """Represents a loaded document."""
    def __init__(
        self,
        content: str,
        metadata: Dict[str, Any],
        source_path: Path
    ):
        self.content = content
        self.metadata = metadata
        self.source_path = source_path
    
    def __repr__(self) -> str:
        return f"Document(source={self.source_path.name}, length={len(self.content)})"


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, file_path: Path) -> Document:
        """Load a document from a file path."""
        pass
    
    @abstractmethod
    def can_load(self, file_path: Path) -> bool:
        """Check if this loader can handle the given file."""
        pass
    
    def load_batch(self, file_paths: List[Path]) -> List[Document]:
        """Load multiple documents."""
        documents = []
        for file_path in file_paths:
            if self.can_load(file_path):
                try:
                    doc = self.load(file_path)
                    documents.append(doc)
                except Exception as e:
                    # Log error but continue
                    print(f"Error loading {file_path}: {e}")
        return documents
