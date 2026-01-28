"""Utilities for loading and using site_common context from retrieved chunks."""

from pathlib import Path
from typing import Dict, Any, Optional, List


class SiteContext:
    """Helper class to load and use site_common context."""
    
    def __init__(self, html_output_dir: Path):
        self.html_output_dir = Path(html_output_dir)
        self.site_common_dir = self.html_output_dir / "site_common"
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def load_site_context(self, chunk_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load site_common context for a chunk based on its metadata.
        
        Args:
            chunk_metadata: Metadata dictionary containing site_common information
            
        Returns:
            Dictionary with site context (header, footer, sidebar, navigation_links, etc.)
        """
        site_common = chunk_metadata.get("site_common")
        if not site_common or not isinstance(site_common, dict):
            return {}
        
        domain_prefix = self._extract_domain_prefix(chunk_metadata)
        if not domain_prefix:
            return {}
        
        if domain_prefix in self._cache:
            return self._cache[domain_prefix]
        
        context = {}
        
        if "header_file" in site_common:
            header_path = self.html_output_dir / site_common["header_file"]
            if header_path.exists():
                try:
                    context["header"] = header_path.read_text(encoding="utf-8")
                except Exception:
                    pass
        
        if "footer_file" in site_common:
            footer_path = self.html_output_dir / site_common["footer_file"]
            if footer_path.exists():
                try:
                    context["footer"] = footer_path.read_text(encoding="utf-8")
                except Exception:
                    pass
        
        if "sidebar_file" in site_common:
            sidebar_path = self.html_output_dir / site_common["sidebar_file"]
            if sidebar_path.exists():
                try:
                    context["sidebar"] = sidebar_path.read_text(encoding="utf-8")
                except Exception:
                    pass
        
        if "metadata_file" in site_common:
            metadata_path = self.html_output_dir / site_common["metadata_file"]
            if metadata_path.exists():
                import json
                try:
                    site_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    context["site_metadata"] = site_metadata
                    nav_links = site_metadata.get("navigation_links", [])
                    if nav_links:
                        context["navigation_links"] = nav_links
                    else:
                        context["navigation_links"] = []
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"Failed to load site metadata from {metadata_path}: {e}")
                    context["navigation_links"] = []
        else:
            context["navigation_links"] = []
        
        self._cache[domain_prefix] = context
        return context
    
    def get_navigation_links(self, chunk_metadata: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get navigation links for a chunk's site."""
        context = self.load_site_context(chunk_metadata)
        return context.get("navigation_links", [])
    
    def get_related_pages(self, chunk_metadata: Dict[str, Any]) -> List[str]:
        """Get URLs of related pages from the same site."""
        nav_links = self.get_navigation_links(chunk_metadata)
        base_url = chunk_metadata.get("url", "")
        if not base_url:
            return []
        
        from urllib.parse import urljoin, urlparse
        parsed = urlparse(base_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        
        related = []
        for link in nav_links:
            href = link.get("href", "")
            if href:
                full_url = urljoin(base, href)
                related.append(full_url)
        
        return related
    
    def enhance_chunk_context(self, chunk_content: str, chunk_metadata: Dict[str, Any]) -> str:
        """Enhance chunk content with site context information."""
        context = self.load_site_context(chunk_metadata)
        
        enhanced_parts = []
        
        if context.get("site_metadata"):
            domain = context["site_metadata"].get("domain", "")
            if domain:
                enhanced_parts.append(f"[Site: {domain}]")
        
        nav_links = context.get("navigation_links", [])
        if nav_links:
            nav_text = ", ".join([link.get("text", "") for link in nav_links[:5]])
            if nav_text:
                enhanced_parts.append(f"[Navigation: {nav_text}]")
        
        enhanced_parts.append(chunk_content)
        
        return "\n\n".join(enhanced_parts)
    
    def _extract_domain_prefix(self, chunk_metadata: Dict[str, Any]) -> Optional[str]:
        """Extract domain prefix from metadata (e.g., 'pdfobject_com')."""
        site_common = chunk_metadata.get("site_common", {})
        if isinstance(site_common, dict):
            header_file = site_common.get("header_file", "")
            if header_file:
                parts = header_file.split("/")
                if len(parts) > 1:
                    filename = parts[-1]
                    prefix = filename.rsplit("_", 1)[0]
                    return prefix
        
        url = chunk_metadata.get("url", "")
        if url:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace(".", "_")
            return domain
        
        return None


def get_site_context_for_chunk(chunk_metadata: Dict[str, Any], html_output_dir: Path) -> Dict[str, Any]:
    """Convenience function to get site context for a chunk."""
    site_context = SiteContext(html_output_dir)
    return site_context.load_site_context(chunk_metadata)
