from typing import List

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings

from app.chunking.base_chunker import BaseChunker, Chunk
from app.loaders.base_loader import Document
from app.config.models.rag_config_model import AzureConfig


class SemanticChunkerStrategy(BaseChunker):
    """Semantic similarity-based chunker."""
    
    def __init__(self, azure_config: AzureConfig, threshold: float = 0.5):
        self.azure_config = azure_config
        self.threshold = threshold
        
        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=azure_config.endpoint,
            azure_deployment=azure_config.embedding_deployment,
            api_key=azure_config.api_key,
            api_version=azure_config.api_version,
            model=azure_config.embedding_model,
            dimensions=azure_config.embedding_dimensions
        )
        
        self.splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=self.threshold
        )
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk document using semantic similarity."""
        texts = self.splitter.split_text(document.content)
        
        chunks = []
        min_chunk_size = 50  # Minimum chunk size to avoid tiny fragments
        for idx, text in enumerate(texts):
            # Skip chunks that are too small
            if len(text.strip()) < min_chunk_size:
                continue
            
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "chunking_strategy": "semantic",
                "threshold": self.threshold,
            })
            
            chunks.append(Chunk(
                content=text,
                metadata=chunk_metadata,
                chunk_index=idx
            ))
        
        return chunks
    
    def get_name(self) -> str:
        return "semantic"
