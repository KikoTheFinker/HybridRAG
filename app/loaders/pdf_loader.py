from pathlib import Path
from typing import Dict, Any

from app.loaders.base_loader import BaseDocumentLoader, Document


class PdfLoader(BaseDocumentLoader):
    """Loader for PDF files."""
    
    def can_load(self, file_path: Path) -> bool:
        """Check if file is a PDF."""
        return file_path.suffix.lower() == ".pdf"
    
    def load(self, file_path: Path) -> Document:
        """Load PDF file."""
        try:
            # Try using PyPDF2 first (lighter weight)
            content, metadata = self._load_with_pypdf2(file_path)
        except Exception:
            try:
                # Fallback to pdfplumber
                content, metadata = self._load_with_pdfplumber(file_path)
            except Exception:
                # Last resort: pymupdf
                content, metadata = self._load_with_pymupdf(file_path)
        
        metadata.update({
            "source_type": "pdf",
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size
        })
        
        return Document(
            content=content,
            metadata=metadata,
            source_path=file_path
        )
    
    def _load_with_pypdf2(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Load PDF using PyPDF2."""
        from PyPDF2 import PdfReader
        
        reader = PdfReader(str(file_path))
        content_parts = []
        metadata = {}
        
        for page in reader.pages:
            content_parts.append(page.extract_text())
        
        if reader.metadata:
            metadata.update({
                "title": reader.metadata.get("/Title"),
                "author": reader.metadata.get("/Author"),
                "subject": reader.metadata.get("/Subject"),
                "creator": reader.metadata.get("/Creator"),
            })
        
        metadata["page_count"] = len(reader.pages)
        
        return "\n\n".join(content_parts), metadata
    
    def _load_with_pdfplumber(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Load PDF using pdfplumber."""
        import pdfplumber
        
        content_parts = []
        metadata = {}
        
        with pdfplumber.open(str(file_path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    content_parts.append(text)
            
            metadata["page_count"] = len(pdf.pages)
        
        return "\n\n".join(content_parts), metadata
    
    def _load_with_pymupdf(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Load PDF using PyMuPDF (fitz)."""
        import fitz  # PyMuPDF
        
        doc = fitz.open(str(file_path))
        content_parts = []
        metadata = {}
        
        for page in doc:
            content_parts.append(page.get_text())
        
        if doc.metadata:
            metadata.update({
                "title": doc.metadata.get("title"),
                "author": doc.metadata.get("author"),
                "subject": doc.metadata.get("subject"),
                "creator": doc.metadata.get("creator"),
            })
        
        metadata["page_count"] = len(doc)
        doc.close()
        
        return "\n\n".join(content_parts), metadata
