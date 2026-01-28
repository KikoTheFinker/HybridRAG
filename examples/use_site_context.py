#!/usr/bin/env python3
"""Example: Using site_common metadata from retrieved chunks."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.rag_orchestrator import RAGOrchestrator
from app.utils.site_context import SiteContext


def main():
    """Demonstrate using site_common metadata."""
    
    orchestrator = RAGOrchestrator()
    query = "what is PDFObject?"
    results = orchestrator.search(query)
    
    if not results:
        print("No results found")
        return
    
    base_path = Path(__file__).resolve().parents[1]
    html_output_dir = base_path / orchestrator.config.data_source.html_output_dir
    site_context = SiteContext(html_output_dir)
    
    print(f"Found {len(results)} results for: {query}\n")
    
    for i, result in enumerate(results[:3], 1):
        print(f"{'='*80}")
        print(f"Result {i}: {result.chunk.metadata.get('title', 'No title')}")
        print(f"{'='*80}")
        
        chunk_metadata = result.chunk.metadata
        context = site_context.load_site_context(chunk_metadata)
        
        nav_links = site_context.get_navigation_links(chunk_metadata)
        if nav_links:
            print("\nNavigation Links:")
            for link in nav_links:
                print(f"  - {link.get('text')}: {link.get('href')}")
        
        related = site_context.get_related_pages(chunk_metadata)
        if related:
            print(f"\nRelated Pages ({len(related)}):")
            for url in related[:5]:
                print(f"  - {url}")
        
        if context.get("site_metadata"):
            site_meta = context["site_metadata"]
            print(f"\nSite Info:")
            print(f"  Domain: {site_meta.get('domain')}")
            print(f"  Has Header: {site_meta.get('has_header')}")
            print(f"  Has Footer: {site_meta.get('has_footer')}")
            print(f"  Has Sidebar: {site_meta.get('has_sidebar')}")
        
        enhanced = site_context.enhance_chunk_context(
            result.chunk.content[:200],
            chunk_metadata
        )
        print(f"\nEnhanced Context Preview:")
        print(f"  {enhanced[:300]}...")
        
        print()


if __name__ == "__main__":
    main()
