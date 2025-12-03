# DevOps and Observability Architecture

This document describes how the AI News RAG Agent is packaged, deployed,
configured, and observed in practice. It is intentionally lightweight so
it can be implemented in a student environment while still following
good production practices.

The key topics are:

- Deployment and runtime environments
- Docker and docker-compose configuration
- Environment and secret management
- Logging and basic metrics via structured logs
- Hooks for future metrics and tracing
- High-level CI/CD outline

## Deployment targets

The service is designed to run in three main modes:

- **Local development (uvicorn)**  
  Run the FastAPI app directly on the host:
  - `uvicorn src.news_rag.api.server:app --reload`

- **Local UI (Streamlit)**  
  Run the Streamlit frontend separately:
  - `streamlit run src/news_rag/ui/streamlit_app.py`

- **Containerized backend (Docker / docker-compose)**  
  Build and run the backend inside a container using the
  `docker/Dockerfile` and `docker/docker-compose.yml`.

At present there is a single container for the backend API. The
Streamlit UI can either run on the host or be containerized separately
in a future iteration.

## Docker image and compose setup

### Dockerfile

The backend image is defined in `docker/Dockerfile` and follows a
simple pattern:

- Base image: `python:3.11-slim`  
  Provides a minimal Python runtime compatible with the project.
- Working directory: `/app`  
  All project files are copied under this directory.
- Dependencies: `requirements.txt` is copied and installed with
  `pip install --no-cache-dir -r requirements.txt`.
- Application code: the entire repo is copied into the image.
- Entrypoint: `uvicorn src.news_rag.api.server:app --host 0.0.0.0 --port 8000`.

This produces a single image that exposes the FastAPI service on port
`8000` inside the container.

### docker-compose

`docker/docker-compose.yml` defines a `backend` service:

- **build.context**: `..` (repository root)  
  Compose builds the image from the top-level project directory.
- **dockerfile**: `docker/Dockerfile`
- **ports**: maps container `8000` to host `8000`.
- **environment**: forwards the four core env vars from the host:
  - `OPENAI_API_KEY`
  - `TAVILY_API_KEY`
  - `GNEWS_API_KEY`
  - `NEWS_RAG_MODEL_NAME`

To run the backend via Compose:

```bash
cd docker
docker-compose up --build
```

This expects the relevant env vars to be set on the host (or provided
via a `.env` file in the Compose project directory).

## Configuration and secrets

Configuration is centralized in `src/news_rag/config.py` through the
`Settings` class:

- `OPENAI_API_KEY`
- `TAVILY_API_KEY`
- `GNEWS_API_KEY`
- `NEWS_RAG_MODEL_NAME`

`Settings` uses `pydantic.BaseSettings`, with `env_file = ".env"`. In a
local workflow:

- `.env.example` documents the required keys.
- Users copy it to `.env` and provide their own secrets.
- The `.env` file **must not** be committed to version control.

When running via Docker Compose, the same variables are read from the
host environment instead of the local `.env` file.

## Logging and basic metrics

### Logging stack

Logging is implemented using **structlog** in
`src/news_rag/logging_config.py`:

- `logging.basicConfig(level=logging.INFO, format="%(message)s")`  
  Sets the base logging level and simple text format for the underlying
  stdlib logger.
- `structlog.configure(...)` wires processors that:
  - Add ISO timestamps.
  - Add log levels.
  - Render stack info and exception information when present.
  - Emit **JSON-formatted log lines** for easy ingestion.

Each module obtains a logger via `get_logger(name)` and emits
structured events with key–value pairs.

### Where logs are emitted

- In **local development**, logs go to stdout in JSON form.
- In **Docker**, logs also go to stdout and can be collected by Docker,
  `docker-compose logs`, or a host-level log shipper (e.g. Fluent Bit,
  Filebeat) into systems like ELK, Loki, or CloudWatch.

### Logged events

The following modules emit structured logs:

- **API layer** (`src/news_rag/api/server.py`):
  - `summarize_request` / `summarize_response`: includes query,
    `time_range`, `verification`, and number of articles.
  - `summarize_error`: summarizes why summarization failed (e.g.,
    missing `OPENAI_API_KEY`).
  - `verification_error`: surfaces issues from the critic pass.
  - `debug_run_graph_request` / `debug_run_graph_response`: captures
    LangGraph invocations and final state (status, search attempts).

- **Retrieval layer** (`src/news_rag/core/retrieval.py`):
  - `retrieve_articles_cache_hit`: indicates cache usage.
  - `retrieve_articles_fetched`: includes which backend was used
    (`tavily` vs `gnews`), topic, `time_range`, and result count.

- **Summarization** (`src/news_rag/core/summarization.py`):
  - `summarize_articles_no_articles`: summary skipped because no
    articles were retrieved.
  - `summarize_articles_call`: topic, article count, and model name
    before calling OpenAI.
  - `summarize_articles_invalid_response` / `non_json` / `missing_keys`:
    diagnose malformed LLM responses.
  - `summarize_articles_success`: topic and number of summary sentences
    produced.

- **Verification** (`src/news_rag/core/verification.py`):
  - `verify_summary_call`: summary topic, article count, and model.
  - `verify_summary_invalid_response` / `non_json` / `missing_keys`:
    diagnose malformed critic responses.
  - `verify_summary_success`: high-level verdict success.

### Using logs as basic metrics

Even without a dedicated metrics backend, these structured logs expose
"metrics-like" signals:

- Count of `/summarize` and `/debug/run-graph` requests over time.
- Distribution of `verification=True/False` usage.
- Backend selection rates for retrieval (`tavily` vs `gnews`).
- Frequency of LLM errors (invalid JSON, missing keys, missing API
  keys).

These can be aggregated by:

- Exporting logs to a JSON-capable log system and building dashboards
  (e.g., Kibana, Loki/Grafana, CloudWatch Logs Insights).
- Running ad-hoc analysis with tools like `jq` or Python notebooks.

## Metrics and tracing hooks (future work)

The codebase is prepared for additional observability without changing
public APIs:

- **Metrics**: add a metrics library (e.g., Prometheus client or
  OpenTelemetry metrics) and increment counters in the same places where
logs are emitted. Examples:
  - `summarize_requests_total{verification="true"}`
  - `summarize_failures_total{reason="missing_api_key"}`
  - `retrieval_backend_used_total{backend="tavily"}`

- **Tracing**: wrap calls to Tavily, GNews, and OpenAI in spans that
  record latency and error information. The LangGraph execution
  (`run_news_agent`) is a natural root span for an end-to-end trace.

These integrations are intentionally left out of the base implementation
to avoid extra dependencies but can be layered on in a production
deployment.

## CI/CD outline

While a full CI/CD pipeline is not provided, the project is structured
to support a simple workflow:

1. **Static checks and formatting** (optional but recommended):
   - `ruff`, `black`, or `flake8` over `src/` and `tests/`.
2. **Unit tests**:
   - `pytest tests/unit` (Tavily/GNews, router, summarization,
     verification).
3. **Integration tests**:
   - `pytest tests/integration` (FastAPI endpoints and end-to-end
     behavior with mocks).
4. **Build Docker image**:
   - `docker build -f docker/Dockerfile -t news-rag-backend .`
5. **Deploy**:
   - Push image to a registry and deploy to the chosen platform
     (container service, VM, or Kubernetes).

Each CI run should at minimum execute the tests and build the Docker
image. Deployment can be manual (e.g., via `docker-compose`) or wired to
an automated stage once the project stabilizes.
