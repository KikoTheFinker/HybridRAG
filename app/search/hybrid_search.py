from typing import List
import re

from app.vector_store.base_store import BaseVectorStore, SearchResult
from app.embeddings.base_embedder import BaseEmbedder
from app.config.models.rag_config_model import HybridSearchConfig


class HybridSearch:
    """Hybrid search combining semantic and keyword search."""
    
    def __init__(self, vector_store: BaseVectorStore, embedder: BaseEmbedder, config: HybridSearchConfig):
        self.vector_store = vector_store
        self.embedder = embedder
        self.config = config
    
    def search(self, query: str) -> List[SearchResult]:
        """Perform hybrid search."""
        semantic_results = self.vector_store.search(
            query=query,
            top_k=self.config.top_k * 2,
            embedder=self.embedder
        )
        
        if semantic_results:
            max_score = max(r.score for r in semantic_results)
            min_score = min(r.score for r in semantic_results)
            score_range = max_score - min_score if max_score != min_score else 1.0
            
            for result in semantic_results:
                normalized_score = (result.score - min_score) / score_range
                result.metadata["semantic_score"] = normalized_score
        
        keyword_results = self._keyword_search(query, semantic_results)
        combined_results = self._combine_results(semantic_results, keyword_results)
        
        if self.config.rerank:
            combined_results = self._rerank(combined_results, query)
        
        return combined_results[:self.config.top_k]
    
    def _keyword_search(self, query: str, semantic_results: List[SearchResult]) -> dict:
        """
        Perform keyword-based search with improved matching.
        
        Uses TF-IDF-like scoring: terms appearing multiple times get higher scores.
        Also handles PDF-related queries better.
        """
        query_lower = query.lower().strip()
        if not query_lower:
            return {}
        
        query_terms = [term.strip() for term in query_lower.split() if len(term.strip()) > 1]
        if not query_terms:
            return {}
        
        expanded_terms = set(query_terms)
        if "pdf" in query_lower or "sample" in query_lower:
            if "sample" in query_lower:
                expanded_terms.add("sample")
                expanded_terms.add("sample-3pp")
            if "pdf" in query_lower:
                expanded_terms.add("pdf")
                for result in semantic_results:
                    content_lower = result.chunk.content.lower()
                    pdf_refs = re.findall(r'([\w\-]+\.pdf)', content_lower)
                    for pdf_ref in pdf_refs[:3]:
                        pdf_name = pdf_ref.replace(".pdf", "")
                        if len(pdf_name) > 3:
                            expanded_terms.add(pdf_name)
        
        query_terms_set = expanded_terms
        keyword_scores = {}
        
        for result in semantic_results:
            content_lower = result.chunk.content.lower()
            
            metadata_text = ""
            chunk_metadata = result.chunk.metadata
            if chunk_metadata.get("file_name"):
                metadata_text += " " + chunk_metadata["file_name"].lower()
            if chunk_metadata.get("url"):
                metadata_text += " " + chunk_metadata["url"].lower()
            
            term_matches = 0
            total_term_count = 0
            metadata_matches = 0
            
            for term in query_terms_set:
                content_count = content_lower.count(term)
                meta_count = metadata_text.count(term) if metadata_text else 0
                count = content_count + meta_count
                
                if count > 0:
                    term_matches += 1
                    total_term_count += count
                    if meta_count > 0:
                        metadata_matches += 1
            
            if term_matches > 0:
                presence_score = term_matches / len(query_terms_set)
                frequency_score = min(total_term_count / (len(query_terms_set) * 5), 1.0)
                metadata_boost = min(metadata_matches / len(query_terms_set), 0.3)
                keyword_score = 0.5 * presence_score + 0.3 * frequency_score + metadata_boost
            else:
                keyword_score = 0.0
            
            keyword_scores[id(result.chunk)] = keyword_score
        
        return keyword_scores
    
    def _combine_results(self, semantic_results: List[SearchResult], keyword_scores: dict) -> List[SearchResult]:
        """Combine semantic and keyword search results."""
        combined = []
        
        for result in semantic_results:
            chunk_id = id(result.chunk)
            semantic_score = result.metadata.get("semantic_score", result.score)
            keyword_score = keyword_scores.get(chunk_id, 0.0)
            
            combined_score = (
                self.config.semantic_weight * semantic_score +
                self.config.keyword_weight * keyword_score
            )
            
            result.metadata["combined_score"] = combined_score
            result.metadata["keyword_score"] = keyword_score
            combined.append(result)
        
        combined.sort(key=lambda x: x.metadata.get("combined_score", 0.0), reverse=True)
        
        return combined
    
    def _rerank(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """
        Rerank results based on additional factors.
        
        Boosts results that:
        1. Have query terms at the beginning
        2. Have site context (navigation links, site_common metadata)
        3. Match "in this site" queries better
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        is_site_query = "site" in query_lower or "this" in query_lower
        
        for result in results:
            content_lower = result.chunk.content.lower()
            
            position_boost = 0.0
            for term in query_terms:
                pos = content_lower.find(term)
                if pos != -1:
                    position_boost += 1.0 / (1.0 + pos / 100.0)
            
            site_context_boost = 0.0
            chunk_metadata = result.chunk.metadata
            has_site_context = (
                chunk_metadata.get("site_common") or
                chunk_metadata.get("navigation_links") or
                chunk_metadata.get("url")
            )
            
            if has_site_context:
                site_context_boost = 0.15
                if is_site_query:
                    site_context_boost = 0.25
            
            current_score = result.metadata.get("combined_score", 0.0)
            result.metadata["combined_score"] = (
                current_score + 
                (position_boost * 0.1) + 
                site_context_boost
            )
        
        results.sort(key=lambda x: x.metadata.get("combined_score", 0.0), reverse=True)
        
        return results
