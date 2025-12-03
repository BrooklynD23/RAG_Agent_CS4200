# Quickstart

This guide walks you from a fresh clone to a running backend and UI.

## 1. Prerequisites

- Python 3.11+
- `git`
- (Optional) Docker and docker-compose for containerized runs

## 2. Clone the repository

```bash
git clone <REPO_URL>
cd RAG_Agent_CS4200
```

## 3. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

If installation fails, ensure you are using a recent Python version and
that build tools are available on your system.

## 5. Configure environment variables

Copy the example env file and fill in your own keys:

```bash
cp .env.example .env
```

Edit `.env` and set:

- `OPENAI_API_KEY` – required for summarization and verification.
- `TAVILY_API_KEY` – required for primary news retrieval.
- `GNEWS_API_KEY` – optional fallback retrieval key.
- `NEWS_RAG_MODEL_NAME` – e.g. `gpt-4o-mini`.

> **Note:** Do **not** commit `.env` to version control.

## 6. Run the FastAPI backend

From the project root:

```bash
uvicorn src.news_rag.api.server:app --reload
```

The API will be available at `http://localhost:8000`.

Useful URLs:

- Health check: `http://localhost:8000/health`
- OpenAPI/Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 7. Run the Streamlit UI

In a separate terminal with the same virtual environment activated:

```bash
streamlit run src/news_rag/ui/streamlit_app.py
```

By default the UI assumes the backend is at `http://localhost:8000`.
You can override this via the `NEWS_RAG_API_BASE_URL` environment
variable.

In the UI you can:

- Enter a topical query (e.g., "Latest developments in solid-state
  batteries").
- Choose a time range (24h, 7d, 30d, all).
- Toggle verification on/off.
- Inspect the generated summary, source list, and debug metadata.

## 8. Optional: run with Docker

If you prefer a containerized backend:

```bash
cd docker
docker-compose up --build
```

This builds the image defined in `docker/Dockerfile` and exposes the
backend on `http://localhost:8000`. Environment variables are forwarded
from your host into the container.

You can still run the Streamlit UI on the host as described above.

## 9. Run tests

To run unit tests:

```bash
pytest tests/unit
```

To run integration tests:

```bash
pytest tests/integration
```

Tests use mocked Tavily, GNews, and OpenAI clients, so they are safe to
run in CI and do not consume external API credits.
