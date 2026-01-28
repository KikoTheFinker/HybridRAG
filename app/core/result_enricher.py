"""Service for enriching search results with additional context."""

from typing import List, Dict, Any

from app.vector_store.base_store import SearchResult
from app.utils.site_context import SiteContext
from app.logging.logger import setup_logger


class ResultEnricher:
    """Enriches search results with site context and additional metadata."""
    
    def __init__(self, site_context: SiteContext):
        self.site_context = site_context
        self.logger = setup_logger(__name__)
    
    def enrich_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Enrich search results with site context and navigation information.
        
        Args:
            results: List of SearchResult objects to enrich
            
        Returns:
            List of enriched SearchResult objects
        """
        enriched_results = []
        enriched_count = 0
        
        for result in results:
            try:
                site_ctx = self.site_context.load_site_context(result.chunk.metadata)
                
                if site_ctx:
                    if "site_context" not in result.metadata:
                        result.metadata["site_context"] = {}
                    
                    result.metadata["site_context"].update({
                        "navigation_links": site_ctx.get("navigation_links", []),
                        "site_metadata": site_ctx.get("site_metadata", {}),
                        "has_header": bool(site_ctx.get("header")),
                        "has_footer": bool(site_ctx.get("footer")),
                        "has_sidebar": bool(site_ctx.get("sidebar")),
                    })
                    
                    nav_links = site_ctx.get("navigation_links", [])
                    if nav_links:
                        result.chunk.metadata["navigation_links"] = nav_links
                        enriched_count += 1
                    
                    try:
                        related_pages = self.site_context.get_related_pages(result.chunk.metadata)
                        if related_pages:
                            result.chunk.metadata["related_pages"] = related_pages
                    except Exception:
                        pass
                
            except Exception as e:
                self.logger.debug(
                    f"Failed to enrich result for chunk "
                    f"{result.chunk.metadata.get('source_path', result.chunk.metadata.get('file_name', 'unknown'))}: {e}"
                )
            
            enriched_results.append(result)
        
        if enriched_count > 0:
            self.logger.debug(f"Enriched {enriched_count} results with site context")
        
        return enriched_results
