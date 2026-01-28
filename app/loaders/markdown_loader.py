import json
from pathlib import Path
from typing import Dict, Any

from app.loaders.base_loader import BaseDocumentLoader, Document


class MarkdownLoader(BaseDocumentLoader):
    """Loader for Markdown files."""
    
    def can_load(self, file_path: Path) -> bool:
        """Check if file is a Markdown file."""
        return file_path.suffix.lower() in [".md", ".markdown"]
    
    def load(self, file_path: Path) -> Document:
        """Load Markdown file."""
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
            "source_type": "markdown",
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "source_path": str(file_path)
        }
        
        # Try to find metadata in html_output/metadata directory
        # Go up from downloads/processed/markdown to find html_output/metadata
        current = file_path.parent
        metadata_dir = None
        
        # Look for html_output/metadata directory
        while current.parent != current:  # Not at root
            html_output = current.parent.parent / "html_output" / "metadata"
            if html_output.exists():
                metadata_dir = html_output
                break
            current = current.parent
        
        # Also try relative to SmartCrawl root
        if not metadata_dir:
            # From downloads/processed/markdown, go to SmartCrawl/html_output/metadata
            smartcrawl_root = file_path.parent.parent.parent.parent
            metadata_dir = smartcrawl_root / "html_output" / "metadata"
        
        if metadata_dir and metadata_dir.exists():
            # Try to find matching metadata file
            # Markdown files might be named like: "sample-3pp.md" 
            # Metadata might be: "pdfobject_com_index.json" or similar
            # Try matching by filename patterns
            md_name = file_path.stem
            
            # Try exact match first
            metadata_file = metadata_dir / (md_name + ".json")
            if not metadata_file.exists():
                # Try with .html.json suffix
                metadata_file = metadata_dir / (md_name + ".html.json")
            
            if metadata_file.exists():
                try:
                    file_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                    base_metadata.update(file_metadata)
                except Exception as e:
                    print(f"Warning: Could not load metadata from {metadata_file}: {e}")
        
        return base_metadata