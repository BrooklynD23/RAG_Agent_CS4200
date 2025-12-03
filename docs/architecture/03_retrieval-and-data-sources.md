# Retrieval and Data Source Architecture

This document describes the Tavily and GNews data sources, caching strategy,
and failure modes for the retrieval layer, adapted from the `Test`
specification.

## 1. Goals

- Retrieve high-quality, up-to-date news articles relevant to a user query.
- Prefer reputable sources and diverse coverage over raw volume.
- Keep external API usage within free-tier limits using caching and fallbacks.

## 2. Primary data source: Tavily

The Tavily Search API is used as the primary news source.

- **Client**: `tavily-python` SDK (`TavilyClient`).
- **Configuration**: `TAVILY_API_KEY` from environment.
- **Usage in code**: `src/news_rag/tools/tavily_tool.py`.

`fetch_news_tavily(topic, max_results, time_range)`:

- Calls `TavilyClient.search(...)` with:
  - `query = topic`.
  - `topic = "news"`.
  - `search_depth = "basic"`.
  - `max_results` as requested.
  - `include_answer = False`.
  - `include_raw_content = True`.
- Maps `response["results"]` into `Article` models:
  - `title`, `url`, `content` or `raw_content`, `score`.
  - `source` derived from URL host.

If `TAVILY_API_KEY` is missing, the function raises a `RuntimeError`, which is
handled by higher layers (e.g., the API) to avoid crashes.

## 3. Fallback data source: GNews

GNews provides a REST API backed by a large corpus of news sources.

- **Endpoint**: `https://gnews.io/api/v4/search`.
- **Configuration**: `GNEWS_API_KEY` from environment (optional).
- **Usage in code**: `src/news_rag/tools/gnews_tool.py` using `httpx`.

`fetch_news_gnews(topic, max_results, time_range)`:

- If `GNEWS_API_KEY` is missing, returns an empty list (graceful no-op).
- Otherwise, calls the search endpoint with params:
  - `q`: topic string.
  - `lang`: currently fixed to `en`.
  - `max`: `max_results`.
  - `apikey`: `GNEWS_API_KEY`.
- Parses JSON and maps `articles` array into `Article` models, including:
  - `source` (from nested `source.name` or URL host).
  - `published_at` (parsed from ISO timestamp where available).
  - `content` or `description` as the text field.

On network or HTTP errors, the function returns an empty list so callers can
degrade gracefully.

## 4. Cache layer

The retrieval layer uses a simple in-memory TTL cache:

- Implementation: `src/news_rag/tools/cache.py`.
- Key: `(query, time_range)`.
- Value: timestamp + list of `Article` objects.
- TTL: 30 minutes.

`retrieve_articles` first checks the cache; if a fresh entry exists, it is
returned immediately. Otherwise, the function calls Tavily, with GNews as a
fallback on error, and then stores the result in the cache.

## 5. Retrieval orchestration

`src/news_rag/core/retrieval.py` defines:

```python
def retrieve_articles(topic: str, time_range: str = "7d", max_results: int = 10) -> List[Article]:
    # 1. cache
    # 2. Tavily
    # 3. GNews fallback
    # 4. cache store
```

This function is used by both the direct `/summarize` API route and the
LangGraph agent nodes.

## 6. Failure modes and fallbacks

Common failure scenarios:

- Tavily API key missing or exhausted.
- GNews API key missing.
- Network errors, timeouts, or non-2xx responses.

Strategies:

- Prefer Tavily; on failure, fall back to GNews.
- If both fail or no API keys are configured, return an empty list and let the
  summarization layer handle the "no articles" case explicitly.
- Cache successful results to reduce repeated external calls for hot topics.

