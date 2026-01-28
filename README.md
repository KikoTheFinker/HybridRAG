# HybridRAG

A production-grade Hybrid Retrieval-Augmented Generation system that processes documents using Azure OpenAI embeddings and multiple chunking strategies.

## Quick Start

### 1. Prerequisites

- Python 3.8+
- Docker (for Qdrant)
- Azure OpenAI account with API key

### 2. Start Qdrant

```powershell
docker-compose up -d
```

This starts Qdrant on `http://localhost:6333` with persistent storage.

### 3. Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. Configure Environment

Copy `.env.example` to `.env` and fill in your Azure OpenAI credentials:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-large
AZURE_EMBEDDING_MODEL=text-embedding-3-large

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=rag-collection
```

### 5. Ingest Documents

```powershell
python scripts/run_rag.py --ingest
```

This processes documents from the configured input directories (see `config/files/rag_config.yaml`).

### 6. Search

```powershell
python scripts/run_rag.py --search "your query here"
```

### 7. Interactive Mode

```powershell
python scripts/run_rag.py --interactive
```

Commands:
- `ingest` - Ingest documents
- `search <query>` - Search for a query
- `clear` - Clear the vector store
- `quit` - Exit

## Features

- **Multiple Chunking Strategies**: Recursive, semantic, and fixed-size chunking
- **Hybrid Search**: Combines semantic (vector) and keyword search with reranking
- **Reranking**: Boosts results based on query term position and site context
- **Site Context**: Automatically enriches results with navigation links and site metadata
- **PDF-to-Markdown Matching**: Intelligently links PDF-derived markdown files to source pages
- **Qdrant Vector Store**: Scalable vector database with full metadata support

## Reranking

Reranking is **enabled by default** (`rerank: true` in config). It boosts results based on:

1. **Position Boost**: Query terms appearing at the beginning of content get higher scores
2. **Site Context Boost**: Results with navigation links and site metadata get boosted (+0.15)
3. **Site Query Boost**: Extra boost (+0.25) for queries containing "site" or "this site"

To disable reranking, set `rerank: false` in `config/files/rag_config.yaml`.

## Configuration

Edit `config/files/rag_config.yaml` to customize:

- **Chunking**: Enable/disable strategies, adjust sizes
- **Hybrid Search**: Adjust semantic/keyword weights, enable/disable reranking
- **Vector Store**: Choose Qdrant or in-memory storage
- **Data Source**: Configure input directories

## Troubleshooting

**Qdrant connection issues:**
- Ensure Docker is running: `docker ps`
- Check Qdrant logs: `docker-compose logs qdrant`

**No documents found:**
- Verify paths in `rag_config.yaml` point to your document directories
- Check that the configured directories contain supported file types (.md, .json, .txt, .html)

**Embedding errors:**
- Verify Azure OpenAI credentials in `.env`
- Check API endpoint and deployment names match your Azure configuration