#!/usr/bin/env python3
"""Main script to run the Hybrid RAG system."""

import argparse
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.rag_orchestrator import RAGOrchestrator
from app.logging.logger import setup_logger


def format_metadata_value(value, indent=2):
    """Format metadata value for display, handling nested structures."""
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = []
        for k, v in value.items():
            formatted = format_metadata_value(v, indent + 2)
            if isinstance(v, (dict, list)) and formatted.startswith("\n"):
                lines.append(f"{' ' * indent}{k}:{formatted}")
            else:
                lines.append(f"{' ' * indent}{k}: {formatted}")
        return "\n" + "\n".join(lines)
    elif isinstance(value, list):
        if not value:
            return "[]"
        lines = []
        for item in value:
            formatted = format_metadata_value(item, indent + 2)
            if isinstance(item, (dict, list)) and formatted.startswith("\n"):
                lines.append(f"{' ' * indent}-{formatted}")
            else:
                lines.append(f"{' ' * indent}- {formatted}")
        return "\n" + "\n".join(lines)
    elif value is None:
        return "null"
    else:
        return str(value)


def print_metadata(metadata, title="Metadata"):
    """Print metadata in a formatted way."""
    print(f"{title}:")
    if not metadata:
        print("  (no metadata)")
        return
    
    for key, value in metadata.items():
        formatted_value = format_metadata_value(value)
        if isinstance(value, (dict, list)) and formatted_value.startswith("\n"):
            print(f"  {key}:{formatted_value}")
        else:
            print(f"  {key}: {formatted_value}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG System")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest documents from configured directories"
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search query"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all data from vector store"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    logger = setup_logger(__name__)
    
    try:
        orchestrator = RAGOrchestrator()
        
        if args.clear:
            orchestrator.clear()
            logger.info("Vector store cleared")
            return
        
        if args.ingest:
            count = orchestrator.ingest_documents()
            logger.info(f"Ingestion complete. Total chunks: {count}")
            return
        
        if args.search:
            results = orchestrator.search(args.search)
            print(f"\nFound {len(results)} results for: {args.search}\n")
            for i, result in enumerate(results, 1):
                print(f"{'='*80}")
                print(f"Result {i} (Score: {result.score:.3f})")
                print(f"{'='*80}")
                
                print_metadata(result.chunk.metadata, "Chunk Metadata")
                
                nav_links = result.chunk.metadata.get("navigation_links", [])
                if nav_links:
                    print("\nNavigation Links:")
                    for link in nav_links:
                        print(f"  - {link.get('text', '')}: {link.get('href', '')}")
                
                related_pages = result.chunk.metadata.get("related_pages", [])
                if related_pages:
                    print(f"\nRelated Pages ({len(related_pages)}):")
                    for url in related_pages[:5]:
                        print(f"  - {url}")
                
                if result.metadata:
                    print()
                    print_metadata(result.metadata, "Search Metadata")
                
                print(f"\nContent ({len(result.chunk.content)} chars):")
                content = result.chunk.content.strip()
                if len(content) > 1000:
                    print(f"  {content[:1000]}...")
                    print(f"  ... ({len(content) - 1000} more characters)")
                else:
                    print(f"  {content}")
                print()
            return
        
        if args.interactive:
            print("Hybrid RAG System - Interactive Mode")
            print("Commands: 'ingest', 'search <query>', 'clear', 'quit'")
            print()
            
            while True:
                try:
                    command = input("RAG> ").strip()
                    
                    if not command:
                        continue
                    
                    if command == "quit" or command == "exit":
                        break
                    
                    if command == "ingest":
                        count = orchestrator.ingest_documents()
                        print(f"Ingestion complete. Total chunks: {count}\n")
                    
                    elif command == "clear":
                        orchestrator.clear()
                        print("Vector store cleared\n")
                    
                    elif command.startswith("search "):
                        query = command[7:].strip()
                        if query:
                            results = orchestrator.search(query)
                            print(f"\nFound {len(results)} results:\n")
                            for i, result in enumerate(results, 1):
                                print(f"{'='*80}")
                                print(f"Result {i} (Score: {result.score:.3f})")
                                print(f"{'='*80}")
                                
                                print_metadata(result.chunk.metadata, "Chunk Metadata")
                                
                                nav_links = result.chunk.metadata.get("navigation_links", [])
                                if nav_links:
                                    print("\nNavigation Links:")
                                    for link in nav_links:
                                        print(f"  - {link.get('text', '')}: {link.get('href', '')}")
                                
                                related_pages = result.chunk.metadata.get("related_pages", [])
                                if related_pages:
                                    print(f"\nRelated Pages ({len(related_pages)}):")
                                    for url in related_pages[:5]:
                                        print(f"  - {url}")
                                
                                if result.metadata:
                                    print()
                                    print_metadata(result.metadata, "Search Metadata")
                                
                                print(f"\nContent ({len(result.chunk.content)} chars):")
                                content = result.chunk.content.strip()
                                if len(content) > 1000:
                                    print(f"  {content[:1000]}...")
                                    print(f"  ... ({len(content) - 1000} more characters)")
                                else:
                                    print(f"  {content}")
                                print()
                        else:
                            print("Please provide a search query\n")
                    
                    else:
                        print(f"Unknown command: {command}\n")
                
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error: {e}\n")
            
            return
        
        # Default: show help
        parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
