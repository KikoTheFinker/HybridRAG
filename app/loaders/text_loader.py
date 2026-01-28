from pathlib import Path
from typing import Dict, Any

from app.loaders.base_loader import BaseDocumentLoader, Document


class TextLoader(BaseDocumentLoader):
    """Loader for plain text files."""
    
    def can_load(self, file_path: Path) -> bool:
        """Check if file is a text file."""
        return file_path.suffix.lower() in [".txt", ".text"]
    
    def load(self, file_path: Path) -> Document:
        """Load text file."""
        content = file_path.read_text(encoding="utf-8")
        
        metadata = {
            "source_type": "text",
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size
        }
        
        return Document(
            content=content,
            metadata=metadata,
            source_path=file_path
        )
