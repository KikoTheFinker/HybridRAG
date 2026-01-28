"""Custom exceptions for RAG system."""


class RAGException(Exception):
    """Base exception for RAG system errors."""
    pass


class DocumentLoadError(RAGException):
    """Raised when document loading fails."""
    pass


class ChunkingError(RAGException):
    """Raised when chunking fails."""
    pass


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(RAGException):
    """Raised when vector store operations fail."""
    pass


class SearchError(RAGException):
    """Raised when search operations fail."""
    pass
