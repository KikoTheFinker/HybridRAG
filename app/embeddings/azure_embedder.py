from typing import List, Optional

from langchain_openai import AzureOpenAIEmbeddings

from app.embeddings.base_embedder import BaseEmbedder
from app.config.models.rag_config_model import AzureConfig


class AzureEmbedder(BaseEmbedder):
    """Azure OpenAI embedding service using text-embedding-large."""
    
    def __init__(self, azure_config: AzureConfig):
        self.azure_config = azure_config
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=azure_config.endpoint,
            azure_deployment=azure_config.embedding_deployment,
            api_key=azure_config.api_key,
            api_version=azure_config.api_version,
            model=azure_config.embedding_model,
            dimensions=azure_config.embedding_dimensions
        )
        
        # Cache dimensions (lazy initialization)
        self._dimensions: Optional[int] = None
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        return self.embeddings.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings with batching."""
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return []
        
        # Use batch embedding for efficiency
        return self.embeddings.embed_documents(valid_texts)
    
    def get_dimensions(self) -> int:
        """Get the dimension of the embeddings (lazy initialization)."""
        if self._dimensions is None:
            # Get dimensions from config or by embedding a test string
            if self.azure_config.embedding_dimensions:
                self._dimensions = self.azure_config.embedding_dimensions
            else:
                # Lazy initialization: embed test string only when needed
                test_embedding = self.embed_text("test")
                self._dimensions = len(test_embedding)
        return self._dimensions
