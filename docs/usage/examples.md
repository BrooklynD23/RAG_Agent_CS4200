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

## 2. Using `/debug/run-graph`

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

## 3. Using the Streamlit UI

Instead of calling the API directly, you can use the UI:

```bash
streamlit run src/news_rag/ui/streamlit_app.py
```

Examples to try in the UI:

- `"Latest developments in solid-state batteries"`
- `"What is happening with global interest rates this week?"`
- `"Recent news about climate change policy in the EU"`

For each query:

- Inspect the main summary under **Summary**.
- Expand individual items under **Sources** to see the raw article
  metadata.
- Open the **Debug info** expander to inspect the `meta` object returned
  by the backend (time range, query type, verification flags, and any
  error messages).

## 4. Capturing outputs for evaluation

When running the manual evaluation protocol (see
`docs/architecture/07_testing-and-evaluation.md`), you can:

- Save JSON responses from `/summarize` to disk.
- Copy screenshots of the Streamlit UI into your project docs.

These artifacts help track summary quality over time as the retrieval
and prompting strategies evolve.
