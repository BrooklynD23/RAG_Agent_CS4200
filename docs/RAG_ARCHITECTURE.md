# RAG News Agent Architecture

This document describes the architecture of the RAG-based news agent system, which replaces the legacy summary-only approach with a full retrieval-augmented generation pipeline.

## Overview

The RAG News Agent is designed to:

1. **Ingest and store** full article content in a vector database
2. **Produce summaries** for initial queries with source citations
3. **Answer follow-up questions** by retrieving relevant chunks from stored articles
4. **Perform web search fallback** when stored sources are insufficient

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface                               │
│                    (Streamlit / API Client)                          │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                               │
│                      POST /rag/query                                 │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LangGraph RAG Pipeline                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Classify   │───▶│ Initial or   │───▶│   Execute    │          │
│  │   Message    │    │  Follow-up?  │    │    Flow      │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  News Fetcher   │    │ Vector Retriever│    │  Web Fallback   │
│  (Tavily/GNews) │    │   (ChromaDB)    │    │   (Tavily)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          ▼                       │                       │
┌─────────────────┐               │                       │
│Article Ingestor │◀──────────────┴───────────────────────┘
│ (Chunk+Embed)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ChromaDB      │
│  Vector Store   │
└─────────────────┘
```

## Core Components

### 1. Models (`src/news_rag/models/`)

#### `rag_state.py`
- **`ArticleChunk`**: Represents a chunk of an article with metadata
- **`RetrievedChunk`**: A chunk retrieved from vector store with similarity score
- **`SourceReference`**: Citation reference for answers
- **`RAGState`**: Full state for the LangGraph pipeline
- **`AgentResponse`**: API response model

### 2. Core Modules (`src/news_rag/core/`)

#### `vector_store.py`
Manages ChromaDB vector database operations:
- `add_chunks()`: Store article chunks with embeddings
- `query_chunks()`: Semantic search with optional conversation filter
- `get_chunks_by_conversation()`: Get all chunks for a conversation
- `delete_conversation_chunks()`: Clean up conversation data

#### `article_ingestor.py`
Handles article processing:
- `chunk_article()`: Split article into chunks with metadata
- `ingest_articles()`: Full pipeline: clean → chunk → embed → store

#### `vector_retriever.py`
High-level retrieval functions:
- `retrieve_relevant_chunks()`: Main retrieval function
- `retrieve_with_context_expansion()`: Include adjacent chunks
- `chunks_to_source_references()`: Convert to citation format
- `format_chunks_for_context()`: Prepare for LLM prompt

#### `sufficiency_checker.py`
Determines if retrieved chunks can answer the question:
- `check_sufficiency_heuristic()`: Fast rule-based check
- `check_sufficiency_llm()`: LLM-based evaluation (optional)

Heuristics used:
- Minimum chunk count (≥2)
- Similarity thresholds (top ≥0.45, avg ≥0.35)
- Content length (≥200 chars)
- Entity coverage
- Temporal relevance

#### `answer_generator.py`
Generates grounded answers:
- `generate_answer()`: For follow-up questions
- `generate_summary_answer()`: For initial queries
- `map_sources_used_to_references()`: Citation mapping

#### `rag_graph.py`
LangGraph state machine orchestrating the pipeline:

**Nodes:**
- `classify_message`: Determine initial vs follow-up
- `fetch_news`: Get articles from web
- `ingest_articles`: Store in vector DB
- `generate_summary`: Create initial summary
- `retrieve_chunks`: Semantic search
- `check_sufficiency`: Evaluate retrieval quality
- `web_search`: Fallback for insufficient data
- `generate_answer`: Create follow-up answer

**Entry Point:**
```python
def run_news_query(
    user_id: Optional[str],
    conversation_id: Optional[str],
    message: str,
    time_range: str = "7d",
    max_articles: int = 10,
    max_chunks: int = 10,
    include_debug: bool = False,
) -> AgentResponse
```

### 3. API (`src/news_rag/api/server.py`)

#### New RAG Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rag/query` | POST | Main query endpoint (initial + follow-up) |
| `/rag/conversation/{id}/sources` | GET | Get sources for a conversation |
| `/rag/conversation/{id}` | DELETE | Clear conversation data |
| `/rag/stats` | GET | Vector store statistics |

#### Request/Response Models

```python
# Request
class RAGQueryRequest:
    message: str
    user_id: Optional[str]
    conversation_id: Optional[str]
    time_range: str = "7d"
    max_articles: int = 10
    max_chunks: int = 10
    include_debug: bool = False

# Response
class RAGQueryResponse:
    answer_text: str
    answer_type: str  # "summary" | "followup_answer" | "web_augmented_answer"
    sources: List[RAGSourceResponse]
    conversation_id: str
    debug: Optional[dict]
```

## Data Flow

### Initial Query Flow

```
User Query
    │
    ▼
Classify as "initial" (no existing chunks)
    │
    ▼
Fetch News (Tavily/GNews)
    │
    ▼
Ingest Articles
    ├── Clean text
    ├── Chunk (RecursiveCharacterTextSplitter)
    ├── Embed (Gemini `text-embedding-004` via Google AI Studio)
    └── Store in ChromaDB with metadata
    │
    ▼
Retrieve Chunks (for summary generation)
    │
    ▼
Generate Summary (with citations)
    │
    ▼
Return AgentResponse
    ├── answer_text: summary
    ├── answer_type: "summary"
    ├── sources: list of articles used
    └── conversation_id: new ID
```

### Follow-up Query Flow (Sufficient Sources)

```
Follow-up Question
    │
    ▼
Classify as "followup" (existing chunks found)
    │
    ▼
Retrieve Relevant Chunks
    │
    ▼
Check Sufficiency → SUFFICIENT
    │
    ▼
Generate Answer (grounded in chunks)
    │
    ▼
Return AgentResponse
    ├── answer_text: answer
    ├── answer_type: "followup_answer"
    └── sources: chunks used
```

### Follow-up Query Flow (Insufficient Sources)

```
Follow-up Question
    │
    ▼
Classify as "followup"
    │
    ▼
Retrieve Relevant Chunks
    │
    ▼
Check Sufficiency → INSUFFICIENT
    │
    ▼
Web Search Fallback
    │
    ▼
Ingest New Articles
    │
    ▼
Re-retrieve Chunks
    │
    ▼
Generate Answer
    │
    ▼
Return AgentResponse
    ├── answer_text: answer
    ├── answer_type: "web_augmented_answer"
    └── sources: original + new sources
```

## Conversation/Topic Tracking

Each conversation is identified by a `conversation_id`:

- **New conversations**: ID is auto-generated
- **Follow-ups**: Pass the same `conversation_id`
- **Article association**: All chunks are tagged with `conversation_id`
- **Scoped retrieval**: Queries filter by `conversation_id`

```python
# First query - new conversation
response1 = run_news_query(message="AI regulations news")
conv_id = response1.conversation_id  # e.g., "a1b2c3d4e5f6"

# Follow-up - same conversation
response2 = run_news_query(
    message="What did the EU propose?",
    conversation_id=conv_id  # Uses same article corpus
)
```

## Configuration

### Environment Variables

```bash
# Required
GOOGLE_API_KEY=your-google-api-key   # Gemini (Google AI Studio)
TAVILY_API_KEY=tvly-...

# Optional
GNEWS_API_KEY=...
NEWS_RAG_MODEL_NAME=gemini-1.5-flash
CHROMA_PERSIST_DIR=.chroma_db
USE_RAG_API=true  # Toggle RAG vs legacy in UI
```

### Settings (`src/news_rag/config.py`)

```python
chunk_size: int = 1000      # Characters per chunk
chunk_overlap: int = 150    # Overlap between chunks
max_articles: int = 10      # Default max articles
```

## Extending the System

### Using a Different Vector Store

1. Create a new module implementing the same interface as `vector_store.py`:
   - `add_chunks(chunks: List[ArticleChunk]) -> int`
   - `query_chunks(query, conversation_id, n_results, threshold) -> List[RetrievedChunk]`
   - `get_chunks_by_conversation(conversation_id) -> List[RetrievedChunk]`
   - `delete_conversation_chunks(conversation_id) -> int`

2. Update imports in `article_ingestor.py`, `vector_retriever.py`, and `rag_graph.py`

### Using a Different News API

1. Create a new tool in `src/news_rag/tools/` following the pattern of `tavily_tool.py`
2. Implement: `fetch_news_xxx(topic, max_results, time_range) -> List[Article]`
3. Update `retrieval.py` to use the new tool

### Adding New Graph Nodes

1. Define the node function in `rag_graph.py`:
   ```python
   def my_new_node(state: RAGState) -> RAGState:
       # Process state
       return state.model_copy(update={...})
   ```

2. Add to the graph:
   ```python
   graph.add_node("my_new_node", my_new_node)
   graph.add_edge("previous_node", "my_new_node")
   ```

## Testing

### Run Unit Tests

```bash
pytest tests/unit/test_rag_pipeline.py -v
```

### Run Integration Tests

```bash
pytest tests/integration/test_rag_api.py -v
```

### Test Coverage

The tests cover:
- ✅ Initial query: news fetch → ingest → summary
- ✅ Follow-up with sufficient sources: retrieve → answer
- ✅ Follow-up with insufficient sources: retrieve → web search → ingest → answer
- ✅ Sufficiency heuristics
- ✅ Chunk processing
- ✅ Source citation mapping
- ✅ API endpoints

## Migration from Legacy System

The legacy `/summarize` endpoint remains available for backward compatibility. To use the new RAG system:

1. **API clients**: Switch from `/summarize` to `/rag/query`
2. **UI**: Set `USE_RAG_API=true` (default)
3. **Conversation tracking**: Store and pass `conversation_id` for follow-ups

### Key Differences

| Feature | Legacy | RAG |
|---------|--------|-----|
| Follow-up answers | Summary text only | Full article retrieval |
| Source access | Not available | Vector DB retrieval |
| Web fallback | Not available | Automatic when needed |
| Conversation tracking | Session state only | Persistent via conversation_id |
| Citations | Summary-level | Chunk-level |
