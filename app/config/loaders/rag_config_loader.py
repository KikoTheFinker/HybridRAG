import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import ValidationError

from app.config.loaders.helpers.yaml_loading_helper import load_yaml
from app.config.models.rag_config_model import RAGConfig, AzureConfig

load_dotenv()


def get_rag_config(config_path: Optional[Path] = None) -> RAGConfig:
    """Load RAG configuration from YAML file and environment variables."""
    if config_path is None:
        # Default to config files directory
        config_path = Path(__file__).resolve().parents[2] / "config" / "files" / "rag_config.yaml"
    
    config_dict = {}
    if config_path.exists():
        config_dict = load_yaml(config_path)
    
    # Override with environment variables if present
    azure_config = {
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", config_dict.get("azure", {}).get("endpoint", "")),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY", config_dict.get("azure", {}).get("api_key", "")),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", config_dict.get("azure", {}).get("api_version", "2024-02-15-preview")),
        "embedding_deployment": os.getenv("AZURE_EMBEDDING_DEPLOYMENT", config_dict.get("azure", {}).get("embedding_deployment", "text-embedding-large")),
        "embedding_model": os.getenv("AZURE_EMBEDDING_MODEL", config_dict.get("azure", {}).get("embedding_model", "text-embedding-3-large")),
    }
    
    if os.getenv("AZURE_EMBEDDING_DIMENSIONS"):
        azure_config["embedding_dimensions"] = int(os.getenv("AZURE_EMBEDDING_DIMENSIONS"))
    
    config_dict["azure"] = azure_config
    
    # Override vector store config with env vars
    if "vector_store" not in config_dict:
        config_dict["vector_store"] = {}
    
    if os.getenv("QDRANT_URL"):
        config_dict["vector_store"]["qdrant_url"] = os.getenv("QDRANT_URL")
    if os.getenv("QDRANT_API_KEY"):
        config_dict["vector_store"]["qdrant_api_key"] = os.getenv("QDRANT_API_KEY")
    if os.getenv("QDRANT_PATH"):
        config_dict["vector_store"]["qdrant_path"] = os.getenv("QDRANT_PATH")
    if os.getenv("QDRANT_COLLECTION_NAME"):
        config_dict["vector_store"]["qdrant_collection_name"] = os.getenv("QDRANT_COLLECTION_NAME")
    
    try:
        return RAGConfig(**config_dict)
    except ValidationError as e:
        raise ValueError(f"Invalid RAG configuration: {e}")
