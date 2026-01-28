from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.vector_store.base_store import BaseVectorStore, SearchResult
from app.chunking.base_chunker import Chunk
from app.embeddings.base_embedder import BaseEmbedder


class InMemoryVectorStore(BaseVectorStore):
    """In-memory vector store implementation (for testing/small datasets)."""
    
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.vectors: List[List[float]] = []
    
    def add_chunks(self, chunks: List[Chunk], embedder: BaseEmbedder) -> None:
        """Add chunks to the in-memory store."""
        texts = [chunk.content for chunk in chunks]
        embeddings = embedder.embed_texts(texts)
        
        self.chunks.extend(chunks)
        self.vectors.extend(embeddings)
    
    def search(self, query: str, top_k: int, embedder: BaseEmbedder) -> List[SearchResult]:
        """Search for similar chunks using cosine similarity."""
        query_vector = embedder.embed_text(query)
        query_vector_array = np.array(query_vector).reshape(1, -1)
        
        # Calculate cosine similarity
        vectors_array = np.array(self.vectors)
        similarities = cosine_similarity(query_vector_array, vectors_array)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include positive similarities
                results.append(SearchResult(
                    chunk=self.chunks[idx],
                    score=float(similarities[idx]),
                    metadata={"similarity": float(similarities[idx])}
                ))
        
        return results
    
    def delete_all(self) -> None:
        """Delete all vectors from the store."""
        self.chunks.clear()
        self.vectors.clear()
    
    def get_count(self) -> int:
        """Get the number of vectors in the store."""
        return len(self.chunks)
