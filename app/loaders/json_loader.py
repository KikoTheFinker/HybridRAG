import json
from pathlib import Path
from typing import Dict, Any

from app.loaders.base_loader import BaseDocumentLoader, Document


class JsonLoader(BaseDocumentLoader):
    """Loader for JSON files."""
    
    def can_load(self, file_path: Path) -> bool:
        """Check if file is a JSON file."""
        return file_path.suffix.lower() == ".json"
    
    def load(self, file_path: Path) -> Document:
        """Load JSON file."""
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            
            # If this is a metadata file, use it as metadata and extract content from referenced files
            if isinstance(data, dict) and "url" in data:
                # This looks like a metadata file - use it as metadata
                metadata = data.copy()
                metadata.update({
                    "source_type": "json_metadata",
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "source_path": str(file_path)
                })
                
                # Convert metadata to readable text content
                content_parts = []
                for key, value in data.items():
                    if key != "site_common" and isinstance(value, str):
                        content_parts.append(f"{key}: {value}")
                    elif key == "site_common" and isinstance(value, dict):
                        content_parts.append(f"Site common files: {', '.join(value.values())}")
                content = "\n".join(content_parts) if content_parts else json.dumps(data, ensure_ascii=False, indent=2)
            else:
                # Regular JSON file - convert to text
                if isinstance(data, dict):
                    content_parts = []
                    for key, value in data.items():
                        if isinstance(value, str):
                            content_parts.append(f"{key}: {value}")
                        elif isinstance(value, (dict, list)):
                            content_parts.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
                    content = "\n".join(content_parts) if content_parts else json.dumps(data, ensure_ascii=False, indent=2)
                elif isinstance(data, list):
                    content = "\n".join([json.dumps(item, ensure_ascii=False) if not isinstance(item, str) else item for item in data])
                else:
                    content = json.dumps(data, ensure_ascii=False, indent=2)
                
                metadata = {
                    "source_type": "json",
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "source_path": str(file_path)
                }
        except Exception:
            # If parsing fails, just read as text
            content = file_path.read_text(encoding="utf-8")
            metadata = {
                "source_type": "json",
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "source_path": str(file_path)
            }
        
        return Document(
            content=content,
            metadata=metadata,
            source_path=file_path
        )
