# Testing and Evaluation Architecture

This document describes how we test the AI News RAG Agent and how to
evaluate the quality and robustness of its outputs.

Goals:

- Catch regressions quickly when changing retrieval, summarization, or
  verification logic.
- Ensure API contracts remain stable.
- Provide a repeatable process for evaluating summary quality and
  safety with real model calls.

## Test pyramid overview

The project uses a simple two-layer pyramid today:

- **Unit tests (`tests/unit`)**
  - Exercise individual components in isolation.
  - Use mocks for external services (Tavily, GNews, OpenAI).

- **Integration tests (`tests/integration`)**
  - Exercise FastAPI endpoints and basic end-to-end behavior.
  - Continue to rely on mocks for external HTTP/LLM calls, so tests are
    deterministic and do not consume API credits.

Load testing and fully automated quality evaluation are out of scope
for this project but can be layered on using the same architecture.

## Unit tests

### Retrieval tools

`tests/unit/test_tavily_tool.py` covers both failure and success paths:

- **API key required**: when `TAVILY_API_KEY` is missing,
  `fetch_news_tavily` raises a `RuntimeError`.  
  This ensures the service fails fast rather than making unauthenticated
  requests.

- **Result mapping**: with a mocked `TavilyClient`, the test asserts
  that Tavily search results are correctly converted to `Article`
  objects (fields: `title`, `url`, `source`, `content`, `score`).

- **GNews fallback**: with a mocked `httpx.get`, the test checks that
  `fetch_news_gnews` maps JSON from the GNews API into `Article`
  instances and handles basic fields like `publishedAt`.

### Router

`tests/unit/test_router.py` verifies that simple time-related phrases
(`"latest"`, `"today"`, year markers, etc.) are classified as
`"news"`. This guards the heuristic routing logic used at the start of
the pipeline.

### Summarization

`tests/unit/test_summarization.py` covers:

- **Input shaping**: `build_summarizer_input` produces the expected
  JSON structure (topic plus `articles[]` with ids and content).
- **API key requirement**: when `OPENAI_API_KEY` is missing,
  `summarize_articles` raises a `RuntimeError` via `_get_openai_client`.
- **Happy path with mocked OpenAI**: a dummy OpenAI client is injected
  that returns a minimal JSON payload. The test asserts that
  `NewsSummary` is built correctly and that original sources are carried
  through.

### Verification

`tests/unit/test_verification.py` mirrors the summarization tests:

- **API key requirement**: missing `OPENAI_API_KEY` raises a
  `RuntimeError`.
- **Happy path with mocked OpenAI**: a dummy client returns a JSON
  object containing `overall_verdict`. The test asserts this field is
  present in the parsed result.

## Integration tests

### FastAPI API

`tests/integration/test_api.py` uses `fastapi.testclient.TestClient` to
exercise the real FastAPI app in-process:

- **`GET /health`**: sanity check that the service is up.

- **`POST /summarize` (success)**: retrieval, summarization, and
  verification are monkeypatched with simple stubs. The test asserts
  that:
  - The endpoint returns HTTP 200.
  - `summary_text` is taken from the stub summary.
  - `meta.verification_result` contains the stubbed verdict.

- **`POST /summarize` (error path)**: summarization is stubbed to raise
  a `RuntimeError` (e.g., missing API key). The test asserts that the
  endpoint still returns HTTP 200 but includes:
  - `meta.error` with the exception message.
  - `sources` populated from the retrieval stub.

- **`POST /debug/run-graph`**: `run_news_agent` is monkeypatched to
  return a synthetic `NewsState`. The test asserts that the endpoint
  returns state with `query` and `status="done"`.

### End-to-end placeholder

`tests/integration/test_end_to_end.py` currently contains a placeholder
test (`assert True`). This marks a clear location to add a true
end-to-end test that:

- Spins up the FastAPI app with real retrieval + summarization +
  verification.
- Uses either the real external services or a more complex fixture
  harness.

## Mocking strategy

The guiding principle is **no real network calls in tests**:

- Tavily and GNews calls are replaced with simple in-memory dummy
  responses.
- OpenAI clients are replaced with tiny classes that mimic the
  `.chat.completions.create(...)` API and return JSON strings.
- FastAPI integration tests monkeypatch the public functions imported in
  `src/news_rag/api/server.py` so that the server itself does not need
  to be modified for testing.

This keeps tests fast, deterministic, and usable in CI without any
secrets.

## Manual evaluation scenarios (P7.3)

Automated tests validate structure and error handling but not summary
quality. For that, we recommend a **manual evaluation protocol** that
can be run periodically or before major changes.

Preconditions:

- Valid `OPENAI_API_KEY` and `TAVILY_API_KEY` in `.env` or environment.
- Backend running locally (uvicorn or Docker).

### Scenario 1 – Breaking news

- Query: `"Latest developments in solid-state batteries"`
- Time range: `"7d"`
- Verification: `True`

Evaluation checklist:

- Are the main recent events covered (announcements, major partnerships,
  breakthroughs)?
- Do citations refer to multiple independent sources?
- Does the critic verdict flag any unsupported or weakly supported
  claims?

### Scenario 2 – Macroeconomics

- Query: `"What is happening with global interest rates this week?"`
- Time range: `"7d"`
- Verification: `True`

Evaluation checklist:

- Are central bank decisions correctly summarized and dated?
- Are geographic scopes clear (e.g., US vs EU vs emerging markets)?
- Are potential hallucinations about non-existent rate decisions
  caught by the critic?

### Scenario 3 – Policy and climate

- Query: `"Recent news about climate change policy in the EU"`
- Time range: `"30d"`
- Verification: `True`

Evaluation checklist:

- Are specific policy names, votes, or regulations correctly linked to
  sources?
- Does the summary differentiate between proposals, passed laws, and
  commentary/opinion pieces?

### Recording results

For each scenario, we recommend capturing:

- The raw JSON response from `/summarize` (including `meta`).
- A short human-written rating (e.g., 1–5) along dimensions like:
  - **Faithfulness** (supported by sources)
  - **Coverage** (important facts included)
  - **Clarity** (readability and structure)

You can paste these into this document under new headings such as:

- `Scenario 1 – Sample Output (2025-12-03)`
- `Scenario 2 – Sample Output (2025-12-03)`

This will turn the document into a living evaluation log. At present,
the scaffolding for scenarios is in place; concrete outputs and
numerical scores should be filled in by the project team as time
permits.
