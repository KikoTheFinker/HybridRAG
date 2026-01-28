from typing import List, Optional
from pydantic import BaseModel, Field


class AzureConfig(BaseModel):
    """Azure OpenAI configuration."""
    endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    api_key: str = Field(..., description="Azure OpenAI API key")
    api_version: str = Field(default="2024-02-15-preview", description="API version")
    embedding_deployment: str = Field(default="text-embedding-large", description="Embedding model deployment name")
    embedding_model: str = Field(default="text-embedding-3-large", description="Embedding model name")
    embedding_dimensions: Optional[int] = Field(default=None, description="Embedding dimensions (None for default)")


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""
    chunk_size: int = Field(default=1000, description="Default chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    use_recursive: bool = Field(default=True, description="Use recursive chunking")
    use_semantic: bool = Field(default=True, description="Use semantic chunking")
    use_fixed: bool = Field(default=True, description="Use fixed-size chunking")
    semantic_threshold: float = Field(default=0.5, description="Semantic similarity threshold")


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    store_type: str = Field(default="qdrant", description="Vector store type: qdrant, in_memory")
    qdrant_url: Optional[str] = Field(default=None, description="Qdrant server URL (for remote instance)")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key (for remote instance)")
    qdrant_path: Optional[str] = Field(default="./qdrant_storage", description="Local path for Qdrant storage")
    qdrant_collection_name: str = Field(default="rag-collection", description="Qdrant collection name")
    qdrant_vector_size: Optional[int] = Field(default=None, description="Vector dimensions (auto-detected from embedder if None)")


class HybridSearchConfig(BaseModel):
    """Hybrid search configuration."""
    semantic_weight: float = Field(default=0.7, description="Weight for semantic search (0-1)")
    keyword_weight: float = Field(default=0.3, description="Weight for keyword search (0-1)")
    top_k: int = Field(default=5, description="Number of results to return")
    rerank: bool = Field(default=True, description="Enable reranking of results")


class DataSourceConfig(BaseModel):
    """Data source configuration."""
    html_output_dir: str = Field(default="../SmartCrawl/html_output", description="Directory containing HTML outputs")
    downloads_dir: str = Field(default="../SmartCrawl/downloads", description="Directory containing processed files")
    include_processed: bool = Field(default=True, description="Include all files from processed directories")
    include_raw_html: bool = Field(default=False, description="Include raw HTML files")


class RAGConfig(BaseModel):
    """Main RAG configuration."""
    azure: AzureConfig
    chunking: ChunkingConfig = ChunkingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    hybrid_search: HybridSearchConfig = HybridSearchConfig()
    data_source: DataSourceConfig = DataSourceConfig()
