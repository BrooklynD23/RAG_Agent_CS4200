# RAG_Agent_CS4200

AI-powered news Retrieval-Augmented Generation (RAG) agent for the CS4200 final project.

## Overview

This system implements a full RAG pipeline for news summarization and question answering:

- **Initial Queries**: Fetches news articles, stores them in a vector database, and generates cited summaries
- **Follow-up Questions**: Retrieves relevant chunks from stored articles to answer questions
- **Web Fallback**: Automatically searches for more sources when stored data is insufficient
- **Citation Tracking**: Every answer includes source citations for verification

### Key Features

| Feature | Description |
|---------|-------------|
| **Vector Storage** | ChromaDB for persistent article chunk storage |
| **Semantic Retrieval** | Gemini embeddings (`text-embedding-004`) via Google AI Studio for relevant chunk retrieval |
| **LangGraph Orchestration** | State machine for complex RAG workflows |
| **Sufficiency Checking** | Heuristic + LLM evaluation of retrieval quality |
| **Web Search Fallback** | Tavily/GNews integration for additional sources |
| **Conversation Tracking** | Persistent conversation IDs for multi-turn interactions |

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── .env.example
├── docs/
│   ├── RAG_ARCHITECTURE.md      # Detailed RAG system documentation
│   ├── architecture/            # Component-level architecture docs
│   ├── usage/                   # Quickstart and examples
│   └── api/                     # API documentation
├── src/news_rag/
│   ├── config.py                # Settings and environment config
│   ├── models/
│   │   ├── news.py              # Article, Summary models
│   │   ├── state.py             # Legacy NewsState
│   │   └── rag_state.py         # RAG state models (NEW)
│   ├── tools/
│   │   ├── tavily_tool.py       # Tavily news search
│   │   ├── gnews_tool.py        # GNews fallback
│   │   └── cache.py             # In-memory caching
│   ├── core/
│   │   ├── graph.py             # Legacy LangGraph agent
│   │   ├── rag_graph.py         # RAG LangGraph pipeline (NEW)
│   │   ├── vector_store.py      # ChromaDB integration (NEW)
│   │   ├── article_ingestor.py  # Chunking & embedding (NEW)
│   │   ├── vector_retriever.py  # Semantic retrieval (NEW)
│   │   ├── sufficiency_checker.py # Retrieval adequacy (NEW)
│   │   ├── answer_generator.py  # Grounded answers (NEW)
│   │   ├── retrieval.py         # News fetching
│   │   ├── summarization.py     # Summary generation
│   │   └── verification.py      # Fact checking
│   ├── api/
│   │   └── server.py            # FastAPI with RAG endpoints
│   └── ui/
│       ├── streamlit_app.py     # Streamlit frontend
│       └── components.py        # UI components
├── tests/
│   ├── unit/
│   │   ├── test_rag_pipeline.py # RAG component tests (NEW)
│   │   └── ...
│   └── integration/
│       ├── test_rag_api.py      # RAG API tests (NEW)
│       └── ...
├── scripts/
│   └── run_app.py               # Launch script with RAG support
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

## Getting Started

### 1. Install dependencies

Create and activate a virtual environment, then run:

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in the values:

```bash
# Required
GOOGLE_API_KEY=your-google-api-key   # Gemini (Google AI Studio) API key
TAVILY_API_KEY=tvly-...

# Optional
GNEWS_API_KEY=...
NEWS_RAG_MODEL_NAME=gemini-1.5-flash
CHROMA_PERSIST_DIR=.chroma_db
USE_RAG_API=true
```

### 3. Run everything with one command

```bash
python scripts/run_app.py
```

The script will:
1. Load environment variables from `.env`
2. Check for required API keys
3. Install/upgrade dependencies (skip with `--skip-install`)
4. Initialize the ChromaDB vector store directory
5. Start the FastAPI backend
6. Start the Streamlit frontend

**Available options:**

| Option | Description |
|--------|-------------|
| `--skip-install` | Skip dependency installation |
| `--upgrade-deps` | Upgrade all packages to latest |
| `--reset-vector-store` | Clear existing vector data |
| `--legacy-mode` | Use legacy API (no RAG) |
| `--backend-port PORT` | Backend port (default: 8000) |
| `--frontend-port PORT` | Frontend port (default: 8501) |
| `--no-reload` | Disable auto-reload |

### 4. Run the API manually

```bash
uvicorn src.news_rag.api.server:app --reload
```

FastAPI docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 5. Run the UI manually

```bash
streamlit run src/news_rag/ui/streamlit_app.py
```

## API Endpoints

### RAG Endpoints (New)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rag/query` | POST | Main query endpoint (initial + follow-up) |
| `/rag/conversation/{id}/sources` | GET | Get sources for a conversation |
| `/rag/conversation/{id}` | DELETE | Clear conversation data |
| `/rag/stats` | GET | Vector store statistics |

### Legacy Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/summarize` | POST | Legacy summarization (no vector storage) |
| `/debug/run-graph` | POST | Debug LangGraph execution |

## Architecture

The RAG pipeline follows this flow:

```
Initial Query → Fetch News → Ingest to Vector DB → Generate Summary
                                    ↓
Follow-up Question → Retrieve Chunks → Check Sufficiency
                                            ↓
                          Sufficient? → Generate Answer
                                ↓
                          Insufficient → Web Search → Ingest → Answer
```

For detailed architecture documentation, see:
- `docs/RAG_ARCHITECTURE.md` - Complete RAG system documentation
- `docs/architecture/` - Component-level architecture docs

## Testing

```bash
# Run all tests
pytest

# Run RAG-specific tests
pytest tests/unit/test_rag_pipeline.py -v
pytest tests/integration/test_rag_api.py -v

# Run with coverage
pytest --cov=src/news_rag
```

## Documentation

| Document | Description |
|----------|-------------|
| `docs/RAG_ARCHITECTURE.md` | Complete RAG pipeline documentation |
| `docs/usage/quickstart.md` | Step-by-step setup guide |
| `docs/usage/examples.md` | API and UI usage examples |
| `docs/api/openapi.md` | API endpoint documentation |
| `docs/architecture/` | Component architecture docs |
