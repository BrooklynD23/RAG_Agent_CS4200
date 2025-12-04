# Project Structure Overview

This document gives a high‑level tour of the news RAG agent project: which
APIs and libraries we use, how the backend and agent are structured, and how a
user query flows end‑to‑end through the system.

---

## 1. Core Idea

The application is an AI‑powered news Retrieval‑Augmented Generation (RAG)
agent. For a natural‑language query about current events, it:

- Retrieves fresh news articles from the open web.
- Summarizes them into a concise, multi‑source answer.
- Attaches explicit citations to each factual claim.
- Optionally runs a verification pass to reduce hallucinations.

All of this is exposed through a FastAPI backend and a Streamlit UI.

---

## 2. Main Technologies and External APIs

### 2.1 Programming language and framework

- **Language:** Python 3.11+
- **Backend:** FastAPI (served with Uvicorn)
- **Frontend/UI:** Streamlit
- **Agent orchestration:** LangChain + LangGraph

### 2.2 External APIs and services

- **OpenAI API**
  - Used for:
    - Query classification (router).
    - News summarization (main writer model).
    - Optional verification/critic pass.
  - Configured via environment variables loaded by `src/news_rag/config.py`
    (`OPENAI_API_KEY`, `NEWS_RAG_MODEL_NAME`).

- **Tavily API**
  - Primary web search and news retrieval tool.
  - Given a query and time range, it returns structured search results
    (titles, URLs, content/snippets, metadata).
  - Configured via `TAVILY_API_KEY` in `.env`.

- **GNews API** (optional fallback)
  - Backup news source if Tavily is unavailable or rate‑limited.
  - Configured via `GNEWS_API_KEY`.

No persistent database or vector store is required; retrieval is ephemeral and
per‑request.

---

## 3. Repository Layout (Conceptual)

Key directories and modules:

- **`src/news_rag/`** – main application package
  - **`config.py`** – loads settings (API keys, model names, tuning params).
  - **`models/`** – Pydantic models representing articles, summaries, and
    agent state.
  - **`tools/`** – wrappers around Tavily, GNews, and caching.
  - **`core/`** – core RAG logic: routing, retrieval, summarization,
    verification, and the LangGraph agent.
  - **`api/server.py`** – FastAPI application and HTTP routes.
  - **`ui/`** – Streamlit app and UI components.

- **`tests/`** – unit and integration tests.
- **`docs/architecture/`** – more detailed architecture write‑ups.
- **`scripts/run_app.py`** – helper to install deps (if needed) and start
  backend + frontend together.

---

## 4. Backend: FastAPI Layer

**File:** `src/news_rag/api/server.py`

FastAPI provides a thin HTTP layer over the agent logic:

- `GET /health`
  - Simple health check returning `{ "status": "ok" }`.

- `POST /summarize`
  - Main production endpoint used by the Streamlit UI.
  - Request body (`SummarizeRequest`):
    - `query: str` – user’s news question.
    - `time_range: str` – e.g. `"24h"`, `"7d"`, `"30d"`, `"all"`.
    - `verification: bool` – enable/disable verification loop.
    - `max_articles: int` – upper bound on how many articles to use.
  - Response:
    - `summary_text` and structured `sentences` with citation data.
    - `sources` – list of article objects with URLs and metadata.
    - `meta` – diagnostic info (query type, time range, errors, etc.).

- `POST /debug/run-graph` (if enabled)
  - Runs the full LangGraph agent and returns internal `NewsState` for
    debugging/teaching.

The FastAPI layer itself is deliberately thin: it validates input, calls into
core functions, and returns a JSON‑serializable result.

---

## 5. Core RAG Components (`src/news_rag/core/`)

This package assembles the retrieval‑augmented generation pipeline.

### 5.1 Router (`router.py`)

- Classifies the user query into a type such as `"news"` vs.
  `"general_knowledge"` using an LLM.
- For this project, we mostly treat all queries as time‑sensitive news
  queries, but the abstraction allows extension to other modes later.

### 5.2 Retrieval (`retrieval.py`)

- Calls **Tavily** as the primary search tool.
- May call **GNews** as a fallback.
- Deduplicates and filters articles.
- Enforces a maximum number of articles and a time range.
- Returns a list of `Article` models (`src/news_rag/models/news.py`).

### 5.3 Summarization (`summarization.py`)

- Uses the **OpenAI API** to generate a **multi‑source, citation‑rich
  summary**:
  - Input: user query + article set.
  - Output: `NewsSummary` model with:
    - `summary_text` – human‑readable summary.
    - `sentences` – per‑sentence objects with `source_ids` that point back to
      source articles.
    - `sources` – the article list.
    - `meta` – details like model name and search statistics.
- Prompts (in `prompts.py`) instruct the model to ground every factual claim
  in the provided sources and to include citations.

### 5.4 Verification (`verification.py`)

- Optional second pass using the LLM as a **critic**:
  - Reads the generated summary and original articles.
  - Checks if each claim is supported by the cited articles.
  - Flags unsupported or underspecified claims.
- Returns a `verification_result` structure (e.g., list of issues) that is
  attached into the `meta` of the final response.

### 5.5 Agent Graph (`graph.py`)

- Uses **LangGraph** to model the process as a stateful agent graph operating
  on `NewsState` (`models/state.py`).
- Main nodes (conceptually):
  - `route_query` → decide query type.
  - `search_news` → call retrieval tools.
  - `grade_results` → decide whether to search again or move on.
  - `summarize_news` → produce the summary.
  - `verify_news` → optional verification step.
  - `handle_error` → terminal error node.
- **Graph overview (simplified)**:
  - `route_query` → `search_news`
  - `search_news` → `grade_results`
  - `grade_results` → `search_news` (if `search_more`)
  - `grade_results` → `summarize_news` (if `summarize`)
  - `grade_results` → `handle_error` (if `fail`)
  - `summarize_news` → `verify_news` → `END` (if verification enabled)
  - `summarize_news` → `END` (if verification disabled)
- Conditional edges control loops like:
  - search → grade → (search more | summarize | fail).
- This separates **control flow** (the graph) from **worker functions**
  (retrieval, summarization, verification).

---

## 6. Configuration and Environment Variables

The application is configured primarily through a `.env` file loaded by
`src/news_rag/config.py` using Pydantic `BaseSettings`:

- **`OPENAI_API_KEY`**
  - Required for all calls to the OpenAI API (routing, summarization,
    verification).
- **`TAVILY_API_KEY`**
  - Required for the Tavily search tool used in retrieval.
- **`GNEWS_API_KEY`** (optional)
  - Enables GNews as a fallback news source if Tavily is unavailable or
    rate‑limited.
- **`NEWS_RAG_MODEL_NAME`**
  - Default LLM name for the news agent (e.g. `gpt-4o-mini`).

The `Settings` class in `config.py` also controls internal tuning values:

- `max_articles` – default maximum number of articles to use in a summary.
- `chunk_size` / `chunk_overlap` – how article text is segmented before
  sending to the LLM.

Additional environment variables used elsewhere:

- **`NEWS_RAG_API_BASE_URL`**
  - Read by the Streamlit UI (`streamlit_app.py`) to know where the FastAPI
    backend lives.
  - Defaults to `http://localhost:8000` if not set.

---

## 7. Tools and Integrations (`src/news_rag/tools/`)

- **`tavily_tool.py`**
  - Wraps Tavily’s HTTP API and exposes a function/tool for the agent.
  - Handles query construction, time ranges, and parsing the response into
    `Article` objects.

- **`gnews_tool.py`**
  - Provides similar functionality for GNews as a fallback.

- **`cache.py`**
  - Optional lightweight caching of search results to reduce API calls and
    latency for repeated queries.

These tools can be used directly in code or bound as tools in a LangChain/
LangGraph agent.

---

## 8. Data Models (`src/news_rag/models/`)

- **`news.py`**
  - `Article` – canonical representation of a news article.
  - `SummarySentence` – a sentence with attached `source_ids`.
  - `NewsSummary` – full structured summary including all sentences, sources,
    and metadata.

- **`state.py`**
  - `NewsState` – state object that flows through the LangGraph agent,
    tracking query, articles, summary, verification status, and errors.

Using Pydantic models gives type safety, validation, and consistent JSON
schemas across the backend and tests.

---

## 9. UI Layer: Streamlit App (`src/news_rag/ui/`)

**File:** `src/news_rag/ui/streamlit_app.py`

The Streamlit UI provides a simple way to interact with the agent:

1. **Configuration sidebar**
   - Time range selector (`24h`, `7d`, `30d`, `all`).
   - Toggle for verification on/off.
   - Slider for maximum number of articles.
   - Reset conversation button.

2. **Main chat‑style interface**
   - User enters a query like “Latest developments in solid‑state
     batteries”.
   - The app sends a `POST /summarize` request to the FastAPI backend using
     `httpx`.
   - Displays:
     - The generated summary.
     - A list of sources (cards with titles, domains, and links).
     - Optional debug metadata in an expandable section.

3. **Follow‑up chat over the existing context**
   - After an initial summary, the UI can use OpenAI again to answer
     follow‑up questions **only based on the retrieved summary and sources**.

The UI reads the backend URL from the `NEWS_RAG_API_BASE_URL` environment
variable (default `http://localhost:8000`).

---

## 10. End‑to‑End Request Flow

Putting it all together, a typical request looks like this:

1. **User enters query in Streamlit**.
2. Streamlit calls **FastAPI** `POST /summarize` with query, time range,
   verification flag, and `max_articles`.
3. FastAPI:
   - Validates the request with Pydantic models.
   - Passes the data to the RAG pipeline / agent.
4. **Router** classifies the query and initializes `NewsState`.
5. **Retrieval** uses Tavily (and possibly GNews + cache) to fetch recent
   news articles.
6. **Summarization** uses OpenAI to write a citation‑aware, multi‑source
   summary.
7. If enabled, **verification** runs a critic pass to evaluate whether the
   summary is supported by the articles.
8. The pipeline produces a `NewsSummary` plus metadata and (optionally)
   verification results.
9. FastAPI serializes this into JSON and returns it to Streamlit.
10. Streamlit renders the summary, sources, and any debug/verification info.

This architecture cleanly separates concerns:

- Tools (Tavily/GNews) vs. agent control flow (LangGraph) vs. presentation
  (Streamlit).
- External API configuration (`config.py` and `.env`) vs. core logic
  (`core/`) vs. transport (`api/server.py`).

The result is a modular, extensible RAG agent that can be reused from both a
UI and programmatic API clients.
