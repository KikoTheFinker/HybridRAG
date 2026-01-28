from typing import List

from app.chunking.base_chunker import BaseChunker, Chunk
from app.chunking.recursive_chunker import RecursiveChunker
from app.chunking.semantic_chunker import SemanticChunkerStrategy
from app.chunking.fixed_chunker import FixedChunker
from app.loaders.base_loader import Document
from app.config.models.rag_config_model import ChunkingConfig, AzureConfig


class ChunkerFactory:
    """Factory for creating and managing chunkers."""
    
    def __init__(self, chunking_config: ChunkingConfig, azure_config: AzureConfig):
        self.chunking_config = chunking_config
        self.azure_config = azure_config
        self.chunkers: List[BaseChunker] = []
        
        self._initialize_chunkers()
    
    def _initialize_chunkers(self):
        """Initialize chunkers based on configuration."""
        if self.chunking_config.use_recursive:
            self.chunkers.append(RecursiveChunker(
                chunk_size=self.chunking_config.chunk_size,
                chunk_overlap=self.chunking_config.chunk_overlap
            ))
        
        if self.chunking_config.use_semantic:
            self.chunkers.append(SemanticChunkerStrategy(
                azure_config=self.azure_config,
                threshold=self.chunking_config.semantic_threshold
            ))
        
        if self.chunking_config.use_fixed:
            self.chunkers.append(FixedChunker(
                chunk_size=self.chunking_config.chunk_size,
                chunk_overlap=self.chunking_config.chunk_overlap
            ))
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Chunk a document using all enabled chunking strategies.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of Chunk objects from all enabled strategies
        """
        all_chunks = []
        min_chunk_size = 50  # Minimum chunk size to avoid tiny fragments
        
        if not document.content or not document.content.strip():
            return []  # Skip empty documents
        
        for chunker in self.chunkers:
            try:
                chunks = chunker.chunk(document)
                # Add chunking strategy name to metadata and filter tiny chunks
                for chunk in chunks:
                    if chunk and chunk.content:
                        chunk.metadata["chunking_strategy"] = chunker.get_name()
                        # Only add chunks that meet minimum size requirement
                        if len(chunk.content.strip()) >= min_chunk_size:
                            all_chunks.append(chunk)
            except Exception as e:
                # Log error but continue with other chunkers
                print(f"Warning: Error chunking with {chunker.get_name()}: {e}")
        
        return all_chunks
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for document in documents:
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)
        return all_chunks
