from app.vector_store.base_store import BaseVectorStore
from app.vector_store.qdrant_store import QdrantVectorStore
from app.vector_store.in_memory_store import InMemoryVectorStore
from app.config.models.rag_config_model import VectorStoreConfig


class VectorStoreFactory:
    """Factory for creating vector stores."""
    
    @staticmethod
    def create(config: VectorStoreConfig) -> BaseVectorStore:
        """Create a vector store based on configuration."""
        if config.store_type == "qdrant":
            return QdrantVectorStore(config)
        elif config.store_type == "in_memory":
            return InMemoryVectorStore()
        else:
            raise ValueError(f"Unknown vector store type: {config.store_type}")
