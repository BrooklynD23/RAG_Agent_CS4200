# API Overview

This project exposes a FastAPI application with two sets of endpoints. All LLM
calls (summaries, answers, verification) use Gemini models via Google AI
Studio, configured by `GOOGLE_API_KEY`.

## RAG Endpoints (Recommended)

- `POST /rag/query` – Main RAG query endpoint for initial queries and follow-ups
- `GET /rag/conversation/{id}/sources` – Get all sources for a conversation
- `DELETE /rag/conversation/{id}` – Clear conversation data from vector store
- `GET /rag/stats` – Vector store statistics

## Legacy Endpoints

- `GET /health` – Health check
- `POST /summarize` – Legacy summarization (no vector storage)
- `POST /debug/run-graph` – Debug LangGraph execution

---

## GET /health

Health probe for readiness/liveness checks.

**Response**

```json
{
  "status": "ok"
}
```

## POST /summarize

Run the retrieval + summarization (+ optional verification) pipeline.

**Request body**

```json
{
  "query": "Latest developments in solid-state batteries",
  "time_range": "7d",
  "verification": true,
  "max_articles": 10
}
```

- `query` (string, required) – User’s news query.
- `time_range` (string, optional, default `"7d"`) – Semantic time horizon for retrieval (currently passed through to tools and cache key).
- `verification` (bool, optional, default `true`) – Whether to run the critic pass after summarization.
- `max_articles` (int, optional, default `10`) – Upper bound on retrieved articles.

**Response body (success, high level)**

```json
{
  "topic": "Latest developments in solid-state batteries",
  "summary_text": "...",
  "sentences": [
    {
      "text": "Company X announced a new prototype battery.",
      "source_ids": ["1", "3"]
    }
  ],
  "sources": [
    {
      "id": "1",
      "title": "Article title",
      "url": "https://example.com/...",
      "source": "example.com",
      "published_at": "2025-12-01T12:00:00Z",
      "content": "...",
      "score": 0.91
    }
  ],
  "meta": {
    "model": "gpt-4o-mini",
    "query_type": "news",
    "time_range": "7d",
    "verification": true,
    "verification_result": {
      "overall_verdict": "accept",
      "issues": []
    }
  }
}
```

If summarization cannot run due to missing `GOOGLE_API_KEY`, the API returns an empty summary but still includes retrieved sources and an error message:

```json
{
  "topic": "...",
  "summary_text": "",
  "sentences": [],
  "sources": [ /* articles */ ],
  "meta": {
    "query_type": "news",
    "time_range": "7d",
    "verification": true,
    "error": "GOOGLE_API_KEY is not configured in the environment."
  }
}
```

**Example curl**

```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
        "query": "Latest developments in solid-state batteries",
        "time_range": "7d",
        "verification": true,
        "max_articles": 10
      }'
```

## POST /debug/run-graph

Execute the LangGraph agent and return the final `NewsState` object.

**Request body**

```json
{
  "query": "Latest developments in solid-state batteries",
  "time_range": "7d",
  "verification": true,
  "max_articles": 10,
  "max_search_attempts": 3
}
```

**Response body (simplified)**

```json
{
  "query": "Latest developments in solid-state batteries",
  "query_type": "news",
  "articles": [ /* Article objects */ ],
  "summary": { /* NewsSummary object */ },
  "search_attempts": 1,
  "max_search_attempts": 3,
  "max_articles": 10,
  "time_range": "7d",
  "verification_enabled": true,
  "verification_result": { /* critic verdict or null */ },
  "status": "done",
  "error": null
}
```

This endpoint is intended for debugging, teaching, and observing the agent’s internal state transitions, not for end-user consumption.

---

# RAG Endpoints

## POST /rag/query

Main entry point for the RAG pipeline. Handles both initial queries and follow-up questions.

**Request body**

```json
{
  "message": "What are the latest AI regulations?",
  "user_id": "optional_user_id",
  "conversation_id": null,
  "time_range": "7d",
  "max_articles": 10,
  "max_chunks": 10,
  "include_debug": false
}
```

- `message` (string, required) – User's query or follow-up question
- `user_id` (string, optional) – User identifier for tracking
- `conversation_id` (string, optional) – Pass existing ID for follow-ups, null for new conversations
- `time_range` (string, optional, default `"7d"`) – Time horizon for news retrieval
- `max_articles` (int, optional, default `10`) – Max articles to fetch
- `max_chunks` (int, optional, default `10`) – Max chunks to retrieve
- `include_debug` (bool, optional, default `false`) – Include debug info in response

**Response body (initial query)**

```json
{
  "answer_text": "Recent developments in AI regulation include...",
  "answer_type": "summary",
  "sources": [
    {
      "article_id": "abc123",
      "url": "https://example.com/article",
      "title": "EU Proposes AI Act",
      "source": "TechNews",
      "published_at": "2024-01-15T10:00:00Z",
      "snippet": "The European Union has proposed..."
    }
  ],
  "conversation_id": "conv_xyz789",
  "debug": null
}
```

**Response body (follow-up)**

```json
{
  "answer_text": "Based on the sources, the EU's proposal includes...",
  "answer_type": "followup_answer",
  "sources": [
    {
      "article_id": "abc123",
      "url": "https://example.com/article",
      "title": "EU Proposes AI Act",
      "source": "TechNews"
    }
  ],
  "conversation_id": "conv_xyz789",
  "debug": {
    "chunks_retrieved": 5,
    "sufficiency": "sufficient",
    "web_search_performed": false
  }
}
```

**Answer types:**

| Type | Description |
|------|-------------|
| `summary` | Initial query summary with citations |
| `followup_answer` | Answer from stored sources |
| `web_augmented_answer` | Answer after web search fallback |
| `error` | Error occurred during processing |

**Example curl (initial query)**

```bash
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
        "message": "What are the latest AI regulations?",
        "time_range": "7d",
        "max_articles": 10
      }'
```

**Example curl (follow-up)**

```bash
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
        "message": "What specific requirements does the EU propose?",
        "conversation_id": "conv_xyz789"
      }'
```

## GET /rag/conversation/{conversation_id}/sources

Get all sources stored for a conversation.

**Path parameters**

- `conversation_id` (string, required) – The conversation ID

**Response body**

```json
{
  "conversation_id": "conv_xyz789",
  "sources": [
    {
      "article_id": "abc123",
      "url": "https://example.com/article",
      "title": "EU Proposes AI Act",
      "source": "TechNews",
      "published_at": "2024-01-15T10:00:00Z"
    }
  ],
  "count": 1
}
```

**Example curl**

```bash
curl "http://localhost:8000/rag/conversation/conv_xyz789/sources"
```

## DELETE /rag/conversation/{conversation_id}

Delete all stored chunks for a conversation.

**Path parameters**

- `conversation_id` (string, required) – The conversation ID

**Response body**

```json
{
  "conversation_id": "conv_xyz789",
  "chunks_deleted": 25,
  "status": "deleted"
}
```

**Example curl**

```bash
curl -X DELETE "http://localhost:8000/rag/conversation/conv_xyz789"
```

## GET /rag/stats

Get vector store statistics.

**Response body**

```json
{
  "status": "ok",
  "vector_store": {
    "name": "news_articles",
    "count": 150,
    "persist_dir": ".chroma_db"
  }
}
```

**Example curl**

```bash
curl "http://localhost:8000/rag/stats"
