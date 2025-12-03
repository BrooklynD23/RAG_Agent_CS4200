# RAG_Agent_CS4200
AI-powered news Retrieval-Augmented Generation (RAG) agent for the CS4200 final project.

The system:
- Accepts topical queries about current events.
- Fetches recent news from the open web via Tavily (with GNews fallback).
- Produces concise summaries grounded in multiple sources.
- Attaches explicit citations to each factual claim via source IDs.
- Optionally runs a verification loop to reduce hallucinations.

This repository implements the architecture described in the `Test` specification
and the markdowns under `docs/architecture`.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── .env.example
├── PROJECT_HANDOFF.md
├── src/
│   └── news_rag/
│       ├── __init__.py
│       ├── config.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── news.py
│       │   └── state.py
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── tavily_tool.py
│       │   ├── gnews_tool.py
│       │   └── cache.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── router.py
│       │   ├── retrieval.py
│       │   ├── summarization.py
│       │   ├── verification.py
│       │   ├── graph.py
│       │   └── prompts.py
│       ├── api/
│       │   ├── __init__.py
│       │   └── server.py
│       └── ui/
│           ├── __init__.py
│           ├── streamlit_app.py
│           └── components.py
├── tests/
│   ├── unit/
│   │   ├── test_tavily_tool.py
│   │   ├── test_router.py
│   │   ├── test_summarization.py
│   │   └── test_verification.py
│   └── integration/
│       ├── test_end_to_end.py
│       └── test_api.py
├── docs/
│   ├── architecture/
│   │   ├── 01_system-overview.md
│   │   ├── 02_backend-agent-architecture.md
│   │   ├── 03_retrieval-and-data-sources.md
│   │   ├── 04_generation-and-prompting.md
│   │   ├── 05_frontend-and-api.md
│   │   ├── 06_devops-and-observability.md
│   │   ├── 07_testing-and-evaluation.md
│   │   └── 08_future-work-and-extensions.md
│   ├── usage/
│   │   ├── quickstart.md
│   │   └── examples.md
│   └── api/
│       └── openapi.md
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

## Getting Started

### 1. Install dependencies

Create and activate a virtual environment, then run:

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in the values:

- `OPENAI_API_KEY`
- `TAVILY_API_KEY`
- `GNEWS_API_KEY` (optional)
- `NEWS_RAG_MODEL_NAME`

### 3. Run the API

```bash
uvicorn src.news_rag.api.server:app --reload
```

FastAPI docs will be available at:

- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`

### 4. Run the UI

```bash
streamlit run src/news_rag/ui/streamlit_app.py
```

The Streamlit app talks to the FastAPI backend (default:
`http://localhost:8000`) and lets you enter queries, configure time
range and verification, and inspect summaries plus sources.

## Architecture Docs

High-level and detailed design docs live in `docs/architecture/` and are
derived from the `Test` specification. Recommended reading order:

1. `docs/architecture/01_system-overview.md`
2. `docs/architecture/02_backend-agent-architecture.md`
3. `docs/architecture/03_retrieval-and-data-sources.md`
4. `docs/architecture/04_generation-and-prompting.md`

## Testing

Run unit tests:

```bash
pytest tests/unit
```

Run integration tests:

```bash
pytest tests/integration
```

Some tests currently assert that unimplemented functions raise
`NotImplementedError` until the full functionality is implemented.

## Usage and Handoff

- For step-by-step setup and run instructions, see
  `docs/usage/quickstart.md`.
- For concrete API and UI examples, see `docs/usage/examples.md`.
- For the phase-by-phase project plan, file scaffolding checklist, and
  session logs, see `PROJECT_HANDOFF.md`.
