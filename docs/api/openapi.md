# API Overview

This project exposes a FastAPI application with the following key endpoints:

- `GET /health` – Health check.
- `POST /summarize` – Main entry point for the news summarization agent.
- `POST /debug/run-graph` – Executes the LangGraph agent and returns the final internal state for debugging.

Below is a concise description of each endpoint, their request/response schemas, and example calls.

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

If summarization cannot run due to missing `OPENAI_API_KEY`, the API returns an empty summary but still includes retrieved sources and an error message:

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
    "error": "OPENAI_API_KEY is not configured in the environment."
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
