import json
from pathlib import Path
from typing import Dict, Any

from app.loaders.base_loader import BaseDocumentLoader, Document


class HtmlLoader(BaseDocumentLoader):
    """Loader for processed HTML text files."""
    
    def can_load(self, file_path: Path) -> bool:
        """Check if file is a processed HTML text file."""
        return file_path.suffix == ".txt" and "processed" in str(file_path)
    
    def load(self, file_path: Path) -> Document:
        """Load processed HTML text file."""
        content = file_path.read_text(encoding="utf-8")
        
        # Try to load corresponding metadata file
        metadata = self._load_metadata(file_path)
        
        return Document(
            content=content,
            metadata=metadata,
            source_path=file_path
        )
    
    def _load_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Load metadata from corresponding JSON file if it exists."""
        base_metadata = {
            "source_type": "html_processed",
            "file_name": file_path.name,
            "source_path": str(file_path)
        }
        
        # Find corresponding metadata file in metadata directory
        # Try multiple possible metadata file names
        metadata_dir = file_path.parent.parent / "metadata"
        
        if metadata_dir.exists():
            # Try exact match: file.txt -> file.json
            metadata_file = metadata_dir / file_path.name.replace(".txt", ".json")
            if not metadata_file.exists():
                # Try: file.txt -> file.html.json (for HTML files that were processed)
                metadata_file = metadata_dir / (file_path.stem + ".html.json")
            if not metadata_file.exists():
                # Try: file.txt -> file (without extension).json
                metadata_file = metadata_dir / (file_path.stem + ".json")
            
            if metadata_file.exists():
                try:
                    file_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                    base_metadata.update(file_metadata)
                except Exception as e:
                    print(f"Warning: Could not load metadata from {metadata_file}: {e}")
        
        return base_metadata


class RawHtmlLoader(BaseDocumentLoader):
    """Loader for raw HTML files."""
    
    def can_load(self, file_path: Path) -> bool:
        """Check if file is raw HTML."""
        return file_path.suffix in [".html", ".htm"] and "raw_html" in str(file_path)
    
    def load(self, file_path: Path) -> Document:
        """Load raw HTML file."""
        from bs4 import BeautifulSoup
        
        html_content = file_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Extract text content
        content = soup.get_text(separator=" ", strip=True)
        
        # Extract metadata
        metadata = {
            "source_type": "html_raw",
            "file_name": file_path.name,
            "title": soup.title.string if soup.title else None,
        }
        
        # Try to load corresponding metadata file
        metadata_file = file_path.parent.parent / "metadata" / file_path.name.replace(".html", ".json").replace(".htm", ".json")
        if metadata_file.exists():
            try:
                file_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                metadata.update(file_metadata)
            except Exception:
                pass
        
        return Document(
            content=content,
            metadata=metadata,
            source_path=file_path
        )
