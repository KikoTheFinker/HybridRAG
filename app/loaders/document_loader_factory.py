from pathlib import Path
from typing import List

from app.loaders.base_loader import BaseDocumentLoader, Document
from app.loaders.html_loader import HtmlLoader, RawHtmlLoader
from app.loaders.markdown_loader import MarkdownLoader
from app.loaders.json_loader import JsonLoader
from app.loaders.text_loader import TextLoader


class DocumentLoaderFactory:
    """Factory for creating and managing document loaders."""
    
    def __init__(self):
        self.loaders: List[BaseDocumentLoader] = [
            MarkdownLoader(),
            JsonLoader(),
            TextLoader(),
            HtmlLoader(),
            RawHtmlLoader(),
        ]
    
    def add_loader(self, loader: BaseDocumentLoader):
        """Add a custom loader."""
        self.loaders.append(loader)
    
    def get_loader(self, file_path: Path) -> BaseDocumentLoader:
        """Get appropriate loader for a file."""
        for loader in self.loaders:
            if loader.can_load(file_path):
                return loader
        raise ValueError(f"No loader found for file: {file_path}")
    
    def load_document(self, file_path: Path) -> Document:
        """Load a document using the appropriate loader."""
        loader = self.get_loader(file_path)
        return loader.load(file_path)
    
    def load_documents(self, file_paths: List[Path]) -> List[Document]:
        """
        Load multiple documents with proper error handling.
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of successfully loaded Document objects
        """
        documents = []
        failed_count = 0
        
        for file_path in file_paths:
            try:
                loader = self.get_loader(file_path)
                doc = loader.load(file_path)
                if doc and doc.content:  # Validate document has content
                    documents.append(doc)
            except ValueError as e:
                # No loader found - skip silently
                failed_count += 1
            except Exception as e:
                # Other errors - log but continue
                failed_count += 1
                print(f"Warning: Failed to load {file_path}: {e}")
        
        if failed_count > 0:
            print(f"Warning: Failed to load {failed_count} out of {len(file_paths)} files")
        
        return documents
