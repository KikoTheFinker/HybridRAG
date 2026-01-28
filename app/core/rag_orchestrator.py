from pathlib import Path
from typing import List, Optional
import re

from app.config.models.rag_config_model import RAGConfig
from app.config.loaders.rag_config_loader import get_rag_config
from app.loaders.document_loader_factory import DocumentLoaderFactory
from app.chunking.chunker_factory import ChunkerFactory
from app.embeddings.azure_embedder import AzureEmbedder
from app.vector_store.store_factory import VectorStoreFactory
from app.search.hybrid_search import HybridSearch
from app.vector_store.base_store import SearchResult
from app.utils.site_context import SiteContext
from app.core.result_enricher import ResultEnricher
from app.logging.logger import setup_logger


class RAGOrchestrator:
    """Main orchestrator for RAG system following SOLID principles."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or get_rag_config()
        self.logger = setup_logger(__name__)
        
        self.base_path = Path(__file__).resolve().parents[2]
        self.html_output_dir = self.base_path / self.config.data_source.html_output_dir
        
        self.document_loader = DocumentLoaderFactory()
        self.chunker_factory = ChunkerFactory(
            chunking_config=self.config.chunking,
            azure_config=self.config.azure
        )
        self.embedder = AzureEmbedder(self.config.azure)
        self.vector_store = VectorStoreFactory.create(self.config.vector_store)
        self.hybrid_search = HybridSearch(
            vector_store=self.vector_store,
            embedder=self.embedder,
            config=self.config.hybrid_search
        )
        
        self.site_context = SiteContext(self.html_output_dir)
        self.result_enricher = ResultEnricher(self.site_context)
        
        self.logger.info("RAG Orchestrator initialized")
    
    def ingest_documents(self, base_path: Optional[Path] = None) -> int:
        """Ingest documents from configured directories."""
        if base_path is None:
            base_path = Path(__file__).resolve().parents[2]
        
        html_output_dir = base_path / self.config.data_source.html_output_dir
        downloads_dir = base_path / self.config.data_source.downloads_dir
        
        all_documents = []
        processed_dirs = []
        
        if html_output_dir.exists():
            html_processed = html_output_dir / "processed"
            if html_processed.exists():
                processed_dirs.append(html_processed)
        
        if downloads_dir.exists():
            downloads_processed = downloads_dir / "processed"
            if downloads_processed.exists():
                for subdir in downloads_processed.iterdir():
                    if subdir.is_dir():
                        processed_dirs.append(subdir)
                processed_dirs.append(downloads_processed)
        
        metadata_map = {}
        url_to_metadata = {}
        pdf_to_metadata = {}
        pdf_reference_counts = {}
        
        if html_output_dir.exists():
            metadata_dir = html_output_dir / "metadata"
            if metadata_dir.exists():
                import json
                import re
                for metadata_file in metadata_dir.glob("*.json"):
                    try:
                        metadata_data = json.loads(metadata_file.read_text(encoding="utf-8"))
                        metadata_stem = metadata_file.stem
                        
                        metadata_map[metadata_stem] = metadata_data
                        
                        if ".html" in metadata_stem:
                            stem_without_html = metadata_stem.replace(".html", "")
                            metadata_map[stem_without_html] = metadata_data
                        
                        if "url" in metadata_data:
                            url = metadata_data["url"]
                            url_to_metadata[url] = metadata_data
                            
                            from urllib.parse import urlparse
                            parsed = urlparse(url)
                            url_path_key = parsed.path.strip("/").replace("/", "_")
                            if url_path_key:
                                metadata_map[url_path_key] = metadata_data
                            
                            path_parts = [p for p in parsed.path.split("/") if p]
                            if path_parts:
                                url_filename = path_parts[-1].replace(".html", "").replace(".htm", "")
                                if url_filename:
                                    metadata_map[url_filename] = metadata_data
                        
                        processed_dir = html_output_dir / "processed"
                        if processed_dir.exists():
                            txt_file = processed_dir / (metadata_stem.replace(".html", "") + ".txt")
                            if not txt_file.exists():
                                txt_file = processed_dir / (metadata_stem + ".txt")
                            
                            if txt_file.exists():
                                try:
                                    content = txt_file.read_text(encoding="utf-8")
                                    pdf_patterns = re.findall(
                                        r'(?:/pdf/|href\s*=\s*["\']?|download\s+)([\w\-]+)\.pdf|([\w\-]+\.pdf)',
                                        content,
                                        re.IGNORECASE
                                    )
                                    
                                    pdf_refs = set()
                                    for match in pdf_patterns:
                                        pdf_ref = match[0] if match[0] else match[1]
                                        if pdf_ref:
                                            pdf_refs.add(pdf_ref.lower())
                                    
                                    simple_pdfs = re.findall(r'\b([\w\-]+)\.pdf\b', content, re.IGNORECASE)
                                    pdf_refs.update([p.lower() for p in simple_pdfs])
                                    
                                    has_download_link = bool(
                                        re.search(r'(?:download|Download).*?sample-3pp|href.*?sample-3pp\.pdf', content, re.IGNORECASE)
                                    )
                                    
                                    for pdf_ref in pdf_refs:
                                        pdf_name = pdf_ref.replace(".pdf", "")
                                        
                                        if pdf_name not in pdf_reference_counts:
                                            pdf_reference_counts[pdf_name] = []
                                        pdf_reference_counts[pdf_name].append({
                                            "metadata": metadata_data,
                                            "count": len([r for r in pdf_refs if r.replace(".pdf", "") == pdf_name]),
                                            "has_download": has_download_link,
                                            "url": metadata_data.get("url", "")
                                        })
                                        
                                        if pdf_name not in pdf_to_metadata:
                                            pdf_to_metadata[pdf_name] = metadata_data
                                        elif has_download_link:
                                            pdf_to_metadata[pdf_name] = metadata_data
                                        
                                        if pdf_ref not in pdf_to_metadata:
                                            pdf_to_metadata[pdf_ref] = metadata_data
                                        elif has_download_link:
                                            pdf_to_metadata[pdf_ref] = metadata_data
                                        
                                        if "_" in pdf_name:
                                            parts = pdf_name.split("_")
                                            if len(parts) > 1 and len(parts[0]) > 8:
                                                pdf_base = "_".join(parts[1:])
                                                if pdf_base not in pdf_to_metadata:
                                                    pdf_to_metadata[pdf_base] = metadata_data
                                                elif has_download_link:
                                                    pdf_to_metadata[pdf_base] = metadata_data
                                            pdf_last = parts[-1]
                                            if pdf_last != pdf_name:
                                                if pdf_last not in pdf_to_metadata:
                                                    pdf_to_metadata[pdf_last] = metadata_data
                                                elif has_download_link:
                                                    pdf_to_metadata[pdf_last] = metadata_data
                                except Exception as e:
                                    self.logger.debug(f"Could not extract PDF refs from {txt_file}: {e}")
                        
                        self.logger.debug(f"Loaded metadata: {metadata_stem} -> {metadata_data.get('url', 'no URL')}")
                    except Exception as e:
                        self.logger.warning(f"Could not load metadata from {metadata_file}: {e}")
        
        for pdf_name, refs in pdf_reference_counts.items():
            if len(refs) > 1:
                refs.sort(key=lambda x: (not x["has_download"], -x["count"]))
                pdf_to_metadata[pdf_name] = refs[0]["metadata"]
                self.logger.debug(
                    f"PDF '{pdf_name}' mapped to {refs[0]['url']} "
                    f"(refs: {refs[0]['count']}, has_download: {refs[0]['has_download']})"
                )
        
        self.logger.info(
            f"Loaded {len(metadata_map)} metadata entries and {len(pdf_to_metadata)} PDF mappings "
            f"from {len(list((html_output_dir / 'metadata').glob('*.json'))) if (html_output_dir / 'metadata').exists() else 0} files"
        )
        
        for processed_dir in processed_dirs:
            all_files = []
            for pattern in ["*.md", "*.markdown", "*.json", "*.txt", "*.html", "*.htm"]:
                all_files.extend(list(processed_dir.glob(pattern)))
            
            if all_files:
                docs = self.document_loader.load_documents(all_files)
                
                matched_count = 0
                for doc in docs:
                    doc_stem = doc.source_path.stem
                    matched = False
                    
                    if doc_stem in metadata_map:
                        doc.metadata.update(metadata_map[doc_stem])
                        matched = True
                        matched_count += 1
                    
                    elif ".html" in doc_stem:
                        stem_without_html = doc_stem.replace(".html", "")
                        if stem_without_html in metadata_map:
                            doc.metadata.update(metadata_map[stem_without_html])
                            matched = True
                            matched_count += 1
                    
                    elif (doc_stem + ".html") in metadata_map:
                        doc.metadata.update(metadata_map[doc_stem + ".html"])
                        matched = True
                        matched_count += 1
                    
                    if not matched and "source_url" in doc.metadata:
                        source_url = doc.metadata.get("source_url")
                        if source_url in url_to_metadata:
                            doc.metadata.update(url_to_metadata[source_url])
                            matched = True
                            matched_count += 1
                    
                    if not matched and doc.metadata.get("source_type") == "markdown":
                        md_stem_clean = doc_stem
                        md_stem_base = re.sub(r'_\d+$', '', md_stem_clean)
                        
                        md_stem_no_hash = md_stem_base
                        if "_" in md_stem_base:
                            parts = md_stem_base.split("_")
                            if len(parts) > 1 and len(parts[0]) > 8:
                                md_stem_no_hash = "_".join(parts[1:])
                            md_stem_last = parts[-1]
                        else:
                            md_stem_last = md_stem_base
                        
                        match_candidates = [
                            md_stem_clean,
                            md_stem_base,
                            md_stem_no_hash,
                            md_stem_last,
                        ]
                        
                        for candidate in match_candidates:
                            if candidate in pdf_to_metadata:
                                doc.metadata.update(pdf_to_metadata[candidate])
                                matched = True
                                matched_count += 1
                                break
                        
                        if not matched:
                            for pdf_name, meta_data in pdf_to_metadata.items():
                                if (md_stem_base in pdf_name or pdf_name in md_stem_base or
                                    md_stem_no_hash in pdf_name or pdf_name in md_stem_no_hash):
                                    doc.metadata.update(meta_data)
                                    matched = True
                                    matched_count += 1
                                    break
                    
                    if not matched:
                        for key, meta_data in metadata_map.items():
                            if doc_stem in key or key in doc_stem:
                                doc.metadata.update(meta_data)
                                matched = True
                                matched_count += 1
                                break
                    
                    if matched:
                        self.logger.debug(f"Matched metadata for {doc.source_path.name}")
                
                all_documents.extend(docs)
                self.logger.info(f"Loaded {len(docs)} documents from {processed_dir.name} ({matched_count} with metadata matched)")
        
        if self.config.data_source.include_raw_html and html_output_dir.exists():
            raw_dir = html_output_dir / "raw_html"
            if raw_dir.exists():
                html_files = list(raw_dir.glob("*.html")) + list(raw_dir.glob("*.htm"))
                docs = self.document_loader.load_documents(html_files)
                all_documents.extend(docs)
                self.logger.info(f"Loaded {len(docs)} raw HTML documents")
        
        if not all_documents:
            self.logger.warning("No documents found to ingest")
            return 0
        
        self.logger.info(f"Chunking {len(all_documents)} documents...")
        chunks = self.chunker_factory.chunk_documents(all_documents)
        
        if not chunks:
            self.logger.warning("No chunks created from documents")
            return 0
        
        self.logger.info(f"Created {len(chunks)} chunks using {len(self.chunker_factory.chunkers)} chunking strategies")
        
        self.logger.info(f"Adding {len(chunks)} chunks to vector store...")
        try:
            self.vector_store.add_chunks(chunks, self.embedder)
            self.logger.info("Chunks added successfully")
        except Exception as e:
            self.logger.error(f"Error adding chunks to vector store: {e}", exc_info=True)
            raise
        
        count = self.vector_store.get_count()
        if count == -1:
            self.logger.warning(
                "Could not retrieve exact count from vector store (likely version mismatch), "
                f"but {len(chunks)} chunks were processed. Collection exists and chunks may have been added."
            )
            # Return the number of chunks we tried to add as an estimate
            return len(chunks)
        else:
            self.logger.info(f"Ingestion complete. Total chunks in store: {count}")
            return count
    
    def search(self, query: str, enrich: bool = True) -> List[SearchResult]:
        """
        Search the RAG system with optional result enrichment.
        
        Args:
            query: Search query string
            enrich: Whether to enrich results with site context (default: True)
        
        Returns:
            List of SearchResult objects, optionally enriched with site context
        """
        self.logger.info(f"Searching for: {query}")
        results = self.hybrid_search.search(query)
        
        if enrich:
            results = self.result_enricher.enrich_results(results)
        
        self.logger.info(f"Found {len(results)} results")
        return results
    
    def clear(self):
        """Clear all data from the vector store."""
        self.logger.info("Clearing vector store...")
        self.vector_store.delete_all()
        self.logger.info("Vector store cleared")
