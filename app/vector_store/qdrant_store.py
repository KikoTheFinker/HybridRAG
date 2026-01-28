from typing import List, Optional
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from qdrant_client.http import models

from app.vector_store.base_store import BaseVectorStore, SearchResult
from app.chunking.base_chunker import Chunk
from app.embeddings.base_embedder import BaseEmbedder
from app.config.models.rag_config_model import VectorStoreConfig


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector store implementation."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.collection_name = config.qdrant_collection_name
        
        if config.qdrant_url:
            self.client = QdrantClient(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key,
            )
        else:
            self.client = QdrantClient(
                path=config.qdrant_path or "./qdrant_storage",
            )
        
        self.vector_size: Optional[int] = config.qdrant_vector_size
    
    def _ensure_collection_exists(self, vector_size: Optional[int] = None):
        """Create the collection if it doesn't exist."""
        try:
            self.client.get_collection(self.collection_name)
            return
        except Exception as e:
            try:
                size = vector_size or self.vector_size or 3072
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=size,
                        distance=Distance.COSINE
                    )
                )
            except Exception as create_error:
                if "already exists" in str(create_error).lower() or "409" in str(create_error):
                    return
                raise
    
    def add_chunks(self, chunks: List[Chunk], embedder: BaseEmbedder) -> None:
        """Add chunks to Qdrant."""
        import logging
        logger = logging.getLogger(__name__)
        
        if not chunks:
            logger.warning("No chunks provided to add_chunks")
            return
        
        logger.info(f"Starting to add {len(chunks)} chunks to Qdrant collection '{self.collection_name}'")
        
        if not self.vector_size:
            try:
                self.vector_size = embedder.get_dimensions()
                logger.info(f"Detected embedding dimensions: {self.vector_size}")
            except Exception as e:
                logger.error(f"Failed to get embedding dimensions: {e}")
                raise
        
        try:
            self._ensure_collection_exists(vector_size=self.vector_size)
            logger.info(f"Collection '{self.collection_name}' verified/created")
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise
        
        texts = [chunk.content for chunk in chunks]
        
        valid_chunks_with_texts = [
            (chunk, text) for chunk, text in zip(chunks, texts)
            if text and text.strip()
        ]
        
        if not valid_chunks_with_texts:
            logger.warning("No valid chunks with content after filtering")
            return
        
        logger.info(f"Filtered to {len(valid_chunks_with_texts)} valid chunks (removed {len(chunks) - len(valid_chunks_with_texts)} empty)")
        
        try:
            valid_texts = [text for _, text in valid_chunks_with_texts]
            logger.info(f"Generating embeddings for {len(valid_texts)} texts...")
            embeddings = embedder.embed_texts(valid_texts)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            if len(embeddings) != len(valid_texts):
                logger.error(f"Embedding count mismatch: {len(embeddings)} embeddings for {len(valid_texts)} texts")
                raise ValueError(f"Expected {len(valid_texts)} embeddings, got {len(embeddings)}")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
        
        points = []
        for (chunk, _), embedding in zip(valid_chunks_with_texts, embeddings):
            if not embedding or len(embedding) != self.vector_size:
                logger.warning(f"Skipping chunk with invalid embedding size: {len(embedding) if embedding else 0} != {self.vector_size}")
                continue
            
            payload = {
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
            }
            
            for key, value in chunk.metadata.items():
                if hasattr(value, '__str__') and hasattr(value, '__fspath__'):
                    payload[key] = str(value)
                elif value is None:
                    payload[key] = None
                elif isinstance(value, (str, int, float, bool, dict, list)):
                    payload[key] = value
                else:
                    payload[key] = str(value)
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            )
            points.append(point)
        
        logger.info(f"Prepared {len(points)} points for upload")
        
        batch_size = 100
        uploaded_count = 0
        failed_batches = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            batch_num = i // batch_size + 1
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                uploaded_count += len(batch)
                if batch_num % 10 == 0:
                    logger.info(f"Uploaded batch {batch_num}/{len(points) // batch_size + 1} ({uploaded_count} points)")
            except Exception as e:
                failed_batches += 1
                logger.error(f"Failed to upload batch {batch_num}: {e}")
                individual_success = 0
                for point in batch:
                    try:
                        self.client.upsert(
                            collection_name=self.collection_name,
                            points=[point]
                        )
                        uploaded_count += 1
                        individual_success += 1
                    except Exception as point_error:
                        logger.debug(f"Failed to upload individual point: {point_error}")
                
                if individual_success > 0:
                    logger.info(f"Recovered {individual_success}/{len(batch)} points from failed batch")
        
        logger.info(f"Upload complete: {uploaded_count}/{len(points)} points uploaded ({failed_batches} batches failed)")
        
        try:
            actual_count = self.get_count()
            logger.info(f"Verified: Collection now contains {actual_count} points")
        except Exception as e:
            logger.warning(f"Could not verify upload count: {e}")
    
    def search(self, query: str, top_k: int, embedder: BaseEmbedder) -> List[SearchResult]:
        """Search for similar chunks."""
        query_vector = embedder.embed_text(query)
        
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        
        results = []
        for result in search_results:
            payload = result.payload
            if not payload:
                continue
            
            content = payload.get("content", "")
            chunk_index = payload.get("chunk_index", 0)
            
            metadata = {}
            for key, value in payload.items():
                if key != "content":
                    metadata[key] = value
            
            chunk = Chunk(
                content=content,
                metadata=metadata,
                chunk_index=chunk_index
            )
            
            score = result.score
            
            results.append(SearchResult(
                chunk=chunk,
                score=float(score),
                metadata={"similarity": float(score)}
            ))
        
        return results
    
    def delete_all(self) -> None:
        """Delete all points from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        
        vector_size = self.vector_size or 3072
        self._ensure_collection_exists(vector_size=vector_size)
    
    def get_count(self) -> int:
        """Get the number of points in the collection."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            count_result = self.client.count_points(
                collection_name=self.collection_name
            )
            count = count_result.count
            logger.debug(f"Collection '{self.collection_name}' contains {count} points")
            return count
        except AttributeError:
            try:
                count_result = self.client.count(
                    collection_name=self.collection_name
                )
                count = count_result.count if hasattr(count_result, 'count') else count_result
                logger.debug(f"Collection '{self.collection_name}' contains {count} points (via count)")
                return count
            except (AttributeError, Exception) as e1:
                logger.warning(f"Using scroll-based count (get_collection/count failed): {e1}")
                try:
                    total = 0
                    offset = None
                    while True:
                        scroll_result = self.client.scroll(
                            collection_name=self.collection_name,
                            limit=1000,
                            offset=offset,
                            with_payload=False,
                            with_vectors=False
                        )
                        points, next_offset = scroll_result
                        total += len(points)
                        if next_offset is None or len(points) == 0:
                            break
                        offset = next_offset
                    logger.info(f"Collection '{self.collection_name}' contains {total} points (via scroll)")
                    return total
                except Exception as e2:
                    logger.error(f"All count methods failed: {e2}")
                    try:
                        collection_info = self.client.get_collection(self.collection_name)
                        count = collection_info.points_count
                        return count
                    except Exception as e3:
                        logger.error(f"get_collection also failed: {e3}")
                        return -1
        except Exception as e:
            logger.error(f"Failed to get collection count: {e}")
            return -1
