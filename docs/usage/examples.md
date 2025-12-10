# Examples

This page shows concrete ways to use the API and UI, plus example
queries that exercise different parts of the system.

Throughout these examples we assume:

- Backend running at `http://localhost:8000`.
- Valid keys set in `.env` or the environment.

## 1. Using the FastAPI `/summarize` endpoint

### Example 1 – Solid-state batteries

Request:

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

Response shape (simplified):

```jsonc
{
  "topic": "Latest developments in solid-state batteries",
  "summary_text": "...",
  "sentences": [
    {"text": "...", "source_ids": ["1", "3"]},
    {"text": "...", "source_ids": ["2"]}
  ],
  "sources": [
    {"id": "1", "title": "...", "url": "...", "source": "..."},
    {"id": "2", "title": "...", "url": "...", "source": "..."}
  ],
  "meta": {
    "query_type": "news",
    "time_range": "7d",
    "verification": true,
    "verification_result": {
      "overall_verdict": "supported",
      "per_sentence": [
        // critic details, if provided by the model
      ]
    }
  }
}
```

What to look for:

- `summary_text` should read as a coherent paragraph-level summary.
- Each `sentences[i].source_ids` should reference valid `sources[j].id`
  entries.
- `meta.verification_result.overall_verdict` gives the critic's overall
  view on faithfulness.

### Example 2 – Macroeconomic news

Request:

```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
        "query": "What is happening with global interest rates this week?",
        "time_range": "7d",
        "verification": true,
        "max_articles": 8
      }'
```

Here you should see a summary that mentions central bank decisions,
market expectations, and relevant geographic regions, with sources from
financial news outlets.

### Example 3 – Turning off verification

If you set `"verification": false`, the response will be the same
schema but `meta.verification_result` will be `null`. This is useful if
you want a faster, cheaper path without the critic.

## 2. Using `/rag/query` (RAG initial + follow-up)

The `/rag/query` endpoint is the recommended way to interact with the RAG
agent. It supports **both**:

- Initial queries that fetch news, ingest articles into the vector store,
  and return a cited summary.
- Follow-up questions that retrieve relevant chunks from stored articles and
  optionally perform an extra Tavily/GNews web search if the existing context
  is insufficient.

### Example 1 – Initial query

```bash
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
        "message": "Latest developments in solid-state batteries",
        "time_range": "7d",
        "max_articles": 10,
        "include_debug": true
      }'
```

Simplified response shape:

```jsonc
{
  "answer_text": "• Bullet-point summary with citations...",
  "answer_type": "summary",
  "sources": [
    {
      "article_id": "1",
      "title": "...",
      "url": "https://example.com/...",
      "source": "example.com",
      "published_at": "2025-12-01T12:00:00Z"
    }
  ],
  "conversation_id": "a1b2c3d4e5f6",
  "debug": {
    "chunks_stored": 6,
    "chunks_retrieved": 5,
    "web_search_performed": false
  }
}
```

The `conversation_id` is important: you pass it back for follow-up questions
so the agent can reuse the same corpus of articles.

### Example 2 – Follow-up question over stored articles

```bash
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
        "message": "Which companies are leading in commercialization?",
        "conversation_id": "a1b2c3d4e5f6",
        "max_chunks": 10,
        "include_debug": true
      }'
```

Possible response (simplified):

```jsonc
{
  "answer_text": "Based on the stored articles, companies A and B are leading...",
  "answer_type": "followup_answer",
  "sources": [ /* subset of articles used */ ],
  "conversation_id": "a1b2c3d4e5f6",
  "debug": {
    "chunks_retrieved": 8,
    "sufficiency": "sufficient",
    "web_search_performed": false
  }
}
```

If the sufficiency checker decides that the existing chunks are **not**
enough, the pipeline will automatically perform another Tavily/GNews search,
ingest the new articles, and then answer with `answer_type="web_augmented_answer"`.

```jsonc
{
  "answer_text": "After pulling in more recent coverage, ...",
  "answer_type": "web_augmented_answer",
  "sources": [ /* original + new articles */ ],
  "conversation_id": "a1b2c3d4e5f6",
  "debug": {
    "chunks_retrieved": 12,
    "sufficiency": "insufficient_initially",
    "web_search_performed": true
  }
}
```

In other words, follow-up questions are grounded in **both** the previously
retrieved articles (from ChromaDB) and any additional articles fetched during
web search fallback.

## 3. Using `/debug/run-graph`

The `/debug/run-graph` endpoint exposes the internal LangGraph state for
debugging.

Example request:

```bash
curl -X POST "http://localhost:8000/debug/run-graph" \
  -H "Content-Type: application/json" \
  -d '{
        "query": "Recent news about climate change policy in the EU",
        "time_range": "30d",
        "verification": true,
        "max_articles": 10,
        "max_search_attempts": 3
      }'
```

Response shape (simplified):

```jsonc
{
  "query": "Recent news about climate change policy in the EU",
  "query_type": "news",
  "articles": [ /* Article objects as in /summarize */ ],
  "summary": { /* NewsSummary object, if summarization succeeded */ },
  "search_attempts": 1,
  "max_search_attempts": 3,
  "max_articles": 10,
  "time_range": "30d",
  "verification_enabled": true,
  "verification_result": { /* critic result, if any */ },
  "status": "done",
  "error": null
}
```

This is particularly helpful when debugging:

- Why retrieval returned few or no articles.
- Whether the graph chose to re-search or proceed to summarization.
- Whether verification ran and how it affected the final state.

## 4. Using the Streamlit UI

Instead of calling the API directly, you can use the UI:

```bash
streamlit run src/news_rag/ui/streamlit_app.py
```

Examples to try in the UI:

- `"Latest developments in solid-state batteries"`
- `"What is happening with global interest rates this week?"`
- `"Recent news about climate change policy in the EU"`

For each query in **RAG mode** (`USE_RAG_API=true`, the default):

- The first message triggers an initial call to `/rag/query` that fetches
  news, ingests articles into ChromaDB, and returns a cited summary.
- The UI stores the returned `conversation_id` and allows you to ask
  follow-up questions in the chat input.
- Follow-up questions are sent back to `/rag/query` with the same
  `conversation_id`, so answers are grounded in both the existing summary and
  the underlying stored articles; if needed, the agent performs **additional
  web search** before answering.
- You can expand **Sources** to see which articles were used, and open
  **Debug info** (when present) to inspect sufficiency checks and whether web
  fallback was triggered.

If you start the UI in **legacy mode** (`USE_RAG_API=false` or
`--legacy-mode` on `scripts/run_app.py`), it will instead use the older
`/summarize` endpoint and restrict follow-up questions to the existing summary
and source list only.

## 5. Capturing outputs for evaluation

When running the manual evaluation protocol (see
`docs/architecture/07_testing-and-evaluation.md`), you can:

- Save JSON responses from `/summarize` to disk.
- Copy screenshots of the Streamlit UI into your project docs.

These artifacts help track summary quality over time as the retrieval
and prompting strategies evolve.
