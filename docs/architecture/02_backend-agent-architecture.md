# Backend + Agent Architecture

This document describes the backend responsibilities, module layout, agent
graph (nodes/edges), and the FastAPI layer for the news RAG agent, adapted from
the `Test` specification.

## 1. Responsibilities

- Provide a clean programmatic API over the news agent.
- Encapsulate:
  - Routing logic (news vs general).
  - Agent graph orchestration (search–grade–write–verify).
  - Tools (Tavily, GNews, cache).
  - Config and secrets management.
- Serve both UI and external API clients.

## 2. Module breakdown

```text
src/news_rag/
  config.py
  models/
    news.py
    state.py
  tools/
    tavily_tool.py
    gnews_tool.py
    cache.py
  core/
    router.py
    retrieval.py
    summarization.py
    verification.py
    graph.py
    prompts.py
  api/
    server.py
```

### 2.1 config.py

`Settings` (Pydantic `BaseSettings`) loads environment variables:

- `OPENAI_API_KEY` or local LLM config.
- `TAVILY_API_KEY`.
- Optional `GNEWS_API_KEY`.

It also holds:

- Default model names (e.g., `NEWS_RAG_MODEL_NAME`).
- Maximum number of articles.
- Chunk sizes and overlap.

### 2.2 models/news.py

Key Pydantic models:

- `Article`:
  - `id: str`
  - `title: str`
  - `url: str`
  - `source: str`
  - `published_at: datetime | None`
  - `content: str` (cleaned body or extended snippet)
  - `score: float | None` (relevance)

- `SummarySentence`:
  - `text: str`
  - `source_ids: list[str]` (references to `Article.id`)

- `NewsSummary`:
  - `topic: str`
  - `summary_text: str`
  - `sentences: list[SummarySentence]`
  - `sources: list[Article]`
  - `meta: dict` (e.g., search stats, model name, warnings)

### 2.3 models/state.py

LangGraph state is represented by `NewsState`:

```python
class NewsState(BaseModel):
    query: str
    query_type: Literal["news", "general"]
    articles: list[Article] = []
    summary: Optional[NewsSummary] = None
    search_attempts: int = 0
    max_search_attempts: int = 3
    max_articles: int = 10
    time_range: str = "7d"
    verification_enabled: bool = True
    verification_result: Optional[dict[str, Any]] = None
    status: Literal["init", "searching", "summarizing", "verifying", "done", "failed"] = "init"
    error: Optional[str] = None
```

The state is passed through LangGraph nodes and mutated/updated at each step.

## 3. Agent graph (LangGraph)

### 3.1 Nodes

- **route_query**
  - Input: `NewsState` with `query`.
  - Output:
    - Sets `query_type` using `classify_query`.
    - Sets `status = "searching"`.

- **search_news**
  - Uses `retrieve_articles` (Tavily + GNews + cache).
  - Increments `search_attempts`.
  - Populates `articles` and keeps `status = "searching"`.

- **grade_results**
  - Lightweight heuristic grading:
    - If `articles` is empty and `search_attempts >= max_search_attempts` →
      mark `status = "failed"`, `error = "no_articles"`.
    - Else, leaves state unchanged.
  - Conditional edge function `_grade_decision` chooses:
    - `"search_more"` → loop back to `search_news`.
    - `"summarize"` → proceed to `summarize_news`.
    - `"fail"` → `handle_error`.

- **summarize_news**
  - Runs `summarize_articles(query, articles)`.
  - Writes `summary` to state.
  - Sets `status = "verifying"` if `verification_enabled` else `"done"`.
  - Conditional edge `_summarize_decision` chooses:
    - `"verify"` → `verify_news`.
    - `"end"` → terminate.

- **verify_news**
  - Uses `verify_summary(summary, articles)`.
  - Writes `verification_result` to state.
  - Sets `status = "done"`.
  - If verification fails due to config (e.g., missing API key), records an
    error in `state.error` but still keeps the summary.

- **handle_error**
  - Terminal node for error conditions (e.g., no articles after max attempts).

### 3.2 Edges

- `route_query → search_news` (always).
- `search_news → grade_results`.
- `grade_results` (conditional):
  - `search_more` → `search_news`.
  - `summarize` → `summarize_news`.
  - `fail` → `handle_error`.
- `summarize_news` (conditional):
  - `verify` → `verify_news`.
  - `end` → `END`.
- `verify_news → END`.
- `handle_error → END`.

## 4. FastAPI layer

`src/news_rag/api/server.py` exposes:

- `GET /health` – basic health check.
- `POST /summarize` – main entrypoint, currently calling retrieval +
  summarization + optional verification directly.
- `POST /debug/run-graph` – executes the LangGraph agent and returns the
  final `NewsState` as JSON for debugging.

`/summarize` is optimized for end-user latency and simpler error handling,
while `/debug/run-graph` is meant for inspection and teaching how the state
evolves through the graph.

## 5. Concurrency and scaling (class project level)

- Single-process `uvicorn` with a modest worker count is sufficient.
- For higher load (out of scope for class):
  - Gunicorn + multiple `uvicorn` workers.
  - Shared model clients across requests (connection pooling).
  - Basic per-IP rate limiting to stay within Tavily/OpenAI quotas.

## 6. Error handling strategy

- **Tool failures** (Tavily/GNews):
  - Wrapped with `try/except`; fall back or degrade gracefully.
- **LLM errors** (OpenAI):
  - Missing API keys raise `RuntimeError` that are surfaced via the `meta`
    field in `/summarize` responses.
- **Agent safeguards**:
  - `max_search_attempts` bounds the number of search loops.
  - LangGraph edges terminate in `END` or `handle_error` to avoid infinite
    loops.

