# AI News RAG Agent – Plan & Handoff

This file is the single source of truth for:
- Project phases and step-by-step plan.
- File scaffolding status.
- Per-session progress logs and next steps.

All agents: **update this file at the end of your session.**

---

## 1. High-Level Phases

- [x] **Phase 0 – Repo & Environment Setup**
- [x] **Phase 1 – Core Scaffolding (packages, config, models)**
- [x] **Phase 2 – Retrieval Layer (Tavily, GNews, cache)**
- [x] **Phase 3 – Summarization & Prompting**
- [x] **Phase 4 – Agent Orchestration (LangGraph)**
- [x] **Phase 5 – API Layer (FastAPI)**
- [x] **Phase 6 – UI Layer (Streamlit)**
- [ ] **Phase 7 – Testing & Evaluation**
- [ ] **Phase 8 – DevOps & Deployment**
- [ ] **Phase 9 – Polish, Docs, and Future Work**

Mark each phase complete when all its tasks are checked off.

---

## 2. Detailed Phase Plan & To-Do List

### Phase 0 – Repo & Environment Setup

**Goal:** Have a clean Python project structure with dependencies installable and env vars configured.

**Tasks**

- [x] P0.1 Create `requirements.txt` or `pyproject.toml` with core deps:
  - `fastapi`, `uvicorn[standard]`, `pydantic`, `httpx`, `streamlit`, `langchain`, `langgraph`, `python-dotenv`, `pytest`
- [x] P0.2 Create `.env.example` with:
  - `OPENAI_API_KEY=`
  - `TAVILY_API_KEY=`
  - `GNEWS_API_KEY=`
  - `NEWS_RAG_MODEL_NAME=`
- [x] P0.3 Verify project installs:
  - `pip install -r requirements.txt`
- [x] P0.4 Document setup in `README.md` (install + run instructions)

### Phase 1 – Core Scaffolding (packages, config, models)

**Goal:** Skeleton package with config and data models wired but not necessarily implemented.

**Tasks**

- [x] P1.1 Create package structure under `src/news_rag/`:
  - [x] `src/news_rag/__init__.py`
  - [x] `src/news_rag/config.py`
  - [x] `src/news_rag/models/__init__.py`
  - [x] `src/news_rag/models/news.py`
  - [x] `src/news_rag/models/state.py`
- [x] P1.2 Implement `Settings` in `config.py` (loads env vars, defaults).
- [x] P1.3 Implement `Article`, `SummarySentence`, `NewsSummary` models.
- [x] P1.4 Implement `NewsState` LangGraph state model.

### Phase 2 – Retrieval Layer (Tavily, GNews, cache)

**Goal:** Given a topic and time range, return a cleaned list of `Article` objects.

**Tasks**

- [x] P2.1 Create tools package:
  - [x] `src/news_rag/tools/__init__.py`
  - [x] `src/news_rag/tools/tavily_tool.py`
  - [x] `src/news_rag/tools/gnews_tool.py`
  - [x] `src/news_rag/tools/cache.py`
- [x] P2.2 Implement `fetch_news_tavily(topic, max_results)` using Tavily API.
- [x] P2.3 Implement `fetch_news_gnews(topic, max_results)` as fallback.
- [x] P2.4 Implement simple in-memory cache in `cache.py`.
- [x] P2.5 Add `src/news_rag/core/retrieval.py`:
  - [x] `retrieve_articles(topic, time_range, max_results)`:
        uses cache → Tavily → fallback → stores in cache.

### Phase 3 – Summarization & Prompting

**Goal:** Turn retrieved articles into a grounded, cited summary.

**Tasks**

- [x] P3.1 Create `src/news_rag/core/prompts.py`:
  - [x] System prompt for summarizer (as in `docs/architecture/04_generation-and-prompting.md`)
  - [x] System prompt for critic / verification agent.
- [x] P3.2 Create `src/news_rag/core/summarization.py`:
  - [x] LLM call that:
        - Accepts topic + article list.
        - Returns `NewsSummary`.
- [x] P3.3 Add JSON parsing / Pydantic validation of LLM output.
- [x] P3.4 Create `src/news_rag/core/verification.py`:
  - [x] Critic pass that evaluates each sentence’s support from sources.

*Note:* Summarization and verification require `OPENAI_API_KEY` to be set in
the environment. When the key is missing, the API returns sources plus a
descriptive error message in `meta.error` instead of failing hard.

### Phase 4 – Agent Orchestration (LangGraph)

**Goal:** Implement the search–grade–write (–verify) loop using LangGraph.

**Tasks**

- [x] P4.1 Implement `src/news_rag/core/router.py`:
  - [x] `classify_query(query) -> "news" | "general"`.
- [x] P4.2 Implement LangGraph wiring in `src/news_rag/core/graph.py`:
  - [x] Nodes: `route_query`, `search_news`, `grade_results`, `summarize_news`, `verify_summary`, `handle_error`.
  - [x] Edges as specified in `02_backend-agent-architecture.md`.
- [x] P4.3 Expose a function like `run_news_agent(query, time_range, verification, max_articles)`.

### Phase 5 – API Layer (FastAPI)

**Goal:** Expose the agent via a simple HTTP API with OpenAPI docs.

**Tasks**

- [x] P5.1 Create `src/news_rag/api/__init__.py` and `src/news_rag/api/server.py`.
- [x] P5.2 Implement endpoints:
  - [x] `GET /health`
  - [x] `POST /summarize` (currently calls retrieval + summarization + optional verification).
  - [x] Optional: `POST /debug/run-graph`.
- [x] P5.3 Document endpoints in `docs/api/openapi.md`.

### Phase 6 – UI Layer (Streamlit)

**Goal:** Simple UI for entering queries and visualizing summary + sources.

**Tasks**

- [x] P6.1 Create `src/news_rag/ui/__init__.py`, `streamlit_app.py`, `components.py`.
- [x] P6.2 Implement basic layout per `05_frontend-and-api.md`.
- [x] P6.3 Wire UI to FastAPI `/summarize` endpoint.

### Phase 7 – Testing & Evaluation

**Goal:** Automated confidence that the system works and does not regress.

**Tasks**

- [x] P7.1 Create tests structure:
  - [x] `tests/unit/test_tavily_tool.py`
  - [x] `tests/unit/test_router.py`
  - [x] `tests/unit/test_summarization.py`
  - [x] `tests/unit/test_verification.py`
  - [x] `tests/integration/test_end_to_end.py`
  - [x] `tests/integration/test_api.py`
- [x] P7.2 Use mocked Tavily/GNews responses for deterministic tests.
- [ ] P7.3 Add evaluation scenarios and record outputs in `docs/architecture/07_testing-and-evaluation.md`.

### Phase 8 – DevOps & Deployment

**Goal:** Containerized, deployable service.

**Tasks**

- [x] P8.1 Create `docker/Dockerfile` and `docker/docker-compose.yml` per `06_devops-and-observability.md`.
- [x] P8.2 Add basic logging and metrics.
- [ ] P8.3 Optionally integrate LangSmith / tracing.

### Phase 9 – Polish, Docs, and Future Work

**Goal:** Clean, documented, presentable project.

**Tasks**

- [ ] P9.1 Fill out `docs/architecture/0x_*.md` from the `Test` spec.
- [x] P9.2 Create `docs/usage/quickstart.md` and `docs/usage/examples.md`.
- [ ] P9.3 Fill `docs/architecture/08_future-work-and-extensions.md`.
- [ ] P9.4 Final README polish and screenshots.

---

## 3. File Scaffolding Checklist

Use this to track which files exist (even if empty).

### Top Level

- [x] `README.md`
- [x] `requirements.txt` or `pyproject.toml`
- [x] `.env.example`
- [x] `PROJECT_HANDOFF.md` (this file)
- [x] `docker/Dockerfile`
- [x] `docker/docker-compose.yml`

### `src/news_rag/`

- [x] `src/news_rag/__init__.py`
- [x] `src/news_rag/config.py`
- [x] `src/news_rag/models/__init__.py`
- [x] `src/news_rag/models/news.py`
- [x] `src/news_rag/models/state.py`
- [x] `src/news_rag/tools/__init__.py`
- [x] `src/news_rag/tools/tavily_tool.py`
- [x] `src/news_rag/tools/gnews_tool.py`
- [x] `src/news_rag/tools/cache.py`
- [x] `src/news_rag/core/__init__.py`
- [x] `src/news_rag/core/router.py`
- [x] `src/news_rag/core/retrieval.py`
- [x] `src/news_rag/core/summarization.py`
- [x] `src/news_rag/core/verification.py`
- [x] `src/news_rag/core/graph.py`
- [x] `src/news_rag/core/prompts.py`
- [x] `src/news_rag/api/__init__.py`
- [x] `src/news_rag/api/server.py`
- [x] `src/news_rag/ui/__init__.py`
- [x] `src/news_rag/ui/streamlit_app.py`
- [x] `src/news_rag/ui/components.py`

### Docs

- [x] `docs/architecture/01_system-overview.md`
- [x] `docs/architecture/02_backend-agent-architecture.md`
- [x] `docs/architecture/03_retrieval-and-data-sources.md`
- [x] `docs/architecture/04_generation-and-prompting.md`
- [x] `docs/architecture/05_frontend-and-api.md`
- [x] `docs/architecture/06_devops-and-observability.md`
- [x] `docs/architecture/07_testing-and-evaluation.md`
- [x] `docs/architecture/08_future-work-and-extensions.md`
- [x] `docs/usage/quickstart.md`
- [x] `docs/usage/examples.md`
- [x] `docs/api/openapi.md`

### Tests

- [x] `tests/unit/test_tavily_tool.py`
- [x] `tests/unit/test_router.py`
- [x] `tests/unit/test_summarization.py`
- [x] `tests/unit/test_verification.py`
- [x] `tests/integration/test_end_to_end.py`
- [x] `tests/integration/test_api.py`

---

## 4. Session Log (Update Every Time)

### Session 2025-12-02 – Agent: Cascade

**Phase focus:**
- Phase 0 – Repo & Environment Setup
- Phase 1 – Core Scaffolding (packages, config, models)

**Tasks completed this session (so far):**
- [x] P0.1 Created `requirements.txt` with core dependencies.
- [x] P0.2 Created `.env.example` with required keys.
- [x] P1.1-1.4 Created `src/news_rag/` package, config, and models skeletons.
- [x] P2.1 Created tools package skeleton (Tavily, GNews, cache).
- [x] P5.1 Created API package skeleton with `/health` and `/summarize` endpoints.
- [x] P6.1 Created UI package skeleton with a basic Streamlit app.
- [x] P7.1 Created tests structure with initial unit and integration tests.
- [x] P8.1 Created `docker/Dockerfile` and `docker/docker-compose.yml`.
- [x] P9.1-9.3 Created docs/usage/api and architecture markdown skeleton files.

**Files created/modified this session:**
- Created: `requirements.txt`
- Created: `.env.example`
- Created: `PROJECT_HANDOFF.md`
- Created: `src/news_rag/__init__.py`
- Created: `src/news_rag/config.py`
- Created: `src/news_rag/models/__init__.py`
- Created: `src/news_rag/models/news.py`
- Created: `src/news_rag/models/state.py`
- Created: `src/news_rag/tools/__init__.py`
- Created: `src/news_rag/tools/tavily_tool.py`
- Created: `src/news_rag/tools/gnews_tool.py`
- Created: `src/news_rag/tools/cache.py`
- Created: `src/news_rag/core/__init__.py`
- Created: `src/news_rag/core/router.py`
- Created: `src/news_rag/core/retrieval.py`
- Created: `src/news_rag/core/summarization.py`
- Created: `src/news_rag/core/verification.py`
- Created: `src/news_rag/core/graph.py`
- Created: `src/news_rag/core/prompts.py`
- Created: `src/news_rag/api/__init__.py`
- Created: `src/news_rag/api/server.py`
- Created: `src/news_rag/ui/__init__.py`
- Created: `src/news_rag/ui/streamlit_app.py`
- Created: `src/news_rag/ui/components.py`
- Created: `tests/unit/test_tavily_tool.py`
- Created: `tests/unit/test_router.py`
- Created: `tests/unit/test_summarization.py`
- Created: `tests/unit/test_verification.py`
- Created: `tests/integration/test_end_to_end.py`
- Created: `tests/integration/test_api.py`
- Created: `docs/architecture/01_system-overview.md`
- Created: `docs/architecture/02_backend-agent-architecture.md`
- Created: `docs/architecture/03_retrieval-and-data-sources.md`
- Created: `docs/architecture/04_generation-and-prompting.md`
- Created: `docs/architecture/05_frontend-and-api.md`
- Created: `docs/architecture/06_devops-and-observability.md`
- Created: `docs/architecture/07_testing-and-evaluation.md`
- Created: `docs/architecture/08_future-work-and-extensions.md`
- Created: `docs/usage/quickstart.md`
- Created: `docs/usage/examples.md`
- Created: `docs/api/openapi.md`
- Created: `docker/Dockerfile`
- Created: `docker/docker-compose.yml`

**Open questions / blockers:**
- None yet – core scaffolding in progress.

**Next recommended actions:**
- Implement retrieval logic for Tavily and GNews tools (P2.2, P2.3, P2.4).
- Implement summarization and verification logic (P3.2, P3.4).
- Implement the LangGraph agent graph and connect it to the API (Phase 4 & 5.2).

---

### Session 2025-12-03 – Agent: Cascade

**Phase focus:**
- Phase 2 – Retrieval Layer
- Phase 3 – Summarization & Prompting
- Phase 4 – Agent Orchestration (LangGraph)
- Phase 5 – API Layer (FastAPI)
- Phase 6 – UI Layer (Streamlit)

**Tasks completed this session (so far):**
- Implemented Tavily + GNews retrieval logic with in-memory TTL cache and fallback.
- Implemented summarization and verification with OpenAI, including prompts, JSON parsing, and Pydantic validation.
- Implemented LangGraph agent graph with `route_query`, `search_news`, `grade_results`, `summarize_news`, `verify_news`, and `handle_error` nodes plus `run_news_agent` helper.
- Wired FastAPI `/summarize` endpoint to retrieval + summarization + optional verification.
- Added `/debug/run-graph` endpoint to expose the LangGraph agent state.
- Wired Streamlit UI to call `/summarize` and render summary, sources, and debug meta.
- Expanded architecture docs (01–03) and API docs (`docs/api/openapi.md`) to match the current implementation.

**Open questions / blockers:**
- Full end-to-end verification depends on user-provided `OPENAI_API_KEY` and `TAVILY_API_KEY`.

**Next recommended actions:**
- Flesh out tests for retrieval, summarization, verification, and API using mocks.
- Add logging/metrics and refine DevOps configuration.
- Complete remaining architecture and usage docs (testing, evaluation, future work).

---

## 5. Current Status

- Phases completed: 0–6 (environment, scaffolding, retrieval, summarization, agent, API, and UI).
- Phases 7–9 in progress (tests, deployment, and polish).
