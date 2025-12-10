# Quickstart

This guide walks you from a fresh clone to a running RAG-powered news agent.

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
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `chromadb` - Vector database for article storage
- `langchain`, `langgraph` - LLM orchestration
- `google-generativeai`, `langchain-google-genai` - Gemini (Google AI Studio) embeddings and chat
- `fastapi`, `uvicorn` - API server
- `streamlit` - Frontend UI

## 5. Configure environment variables

Copy the example env file and fill in your own keys:

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
# Required
GOOGLE_API_KEY=your-google-api-key   # For Gemini embeddings, summaries, and answers
TAVILY_API_KEY=tvly-...              # For news retrieval

# Optional
GNEWS_API_KEY=...                    # Fallback news source
NEWS_RAG_MODEL_NAME=gemini-1.5-flash
CHROMA_PERSIST_DIR=.chroma_db        # Vector store location
USE_RAG_API=true                     # Enable RAG mode (default)
```

> **Note:** Do **not** commit `.env` to version control.

## 6. Quick Start (Recommended)

The easiest way to run everything:

```bash
python scripts/run_app.py
```

This will:
1. Load your `.env` file
2. Check for required API keys
3. Install dependencies if needed
4. Initialize the ChromaDB vector store
5. Start the FastAPI backend on port 8000
6. Start the Streamlit UI on port 8501

**Useful options:**

```bash
# Reset vector store (clear all stored articles)
python scripts/run_app.py --reset-vector-store

# Use legacy mode (no RAG, summary-only)
python scripts/run_app.py --legacy-mode

# Skip dependency installation
python scripts/run_app.py --skip-install

# Custom ports
python scripts/run_app.py --backend-port 8080 --frontend-port 8502
```

## 7. Manual Setup

### Run the FastAPI backend

```bash
uvicorn src.news_rag.api.server:app --reload
```

The API will be available at `http://localhost:8000`.

**Key endpoints:**

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `POST /rag/query` | Main RAG query endpoint |
| `GET /rag/conversation/{id}/sources` | Get conversation sources |
| `DELETE /rag/conversation/{id}` | Clear conversation |
| `GET /rag/stats` | Vector store statistics |
| `POST /summarize` | Legacy summarization |

**Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Run the Streamlit UI

In a separate terminal:

```bash
streamlit run src/news_rag/ui/streamlit_app.py
```

The UI will connect to the backend at `http://localhost:8000` by default.
Override with `NEWS_RAG_API_BASE_URL` environment variable.

## 8. Using the Application

### Initial Query

1. Enter a news topic (e.g., "Latest developments in AI regulation")
2. The system will:
   - Fetch relevant news articles
   - Store them in the vector database
   - Generate a summary with citations

### Follow-up Questions

1. After an initial query, ask follow-up questions
2. The system will:
   - Retrieve relevant chunks from stored articles
   - Check if the information is sufficient
   - If insufficient, automatically search for more sources
   - Generate an answer with citations

### Conversation Management

- **Reset**: Click "Reset conversation" in the sidebar to start fresh
- **Conversation ID**: Displayed in the sidebar for debugging
- **Sources**: Expandable section shows which articles were used

## 9. Docker Setup (Optional)

```bash
cd docker
docker-compose up --build
```

This runs the backend on `http://localhost:8000`. The Streamlit UI can
still run on the host.

## 10. Run Tests

```bash
# All tests
pytest

# RAG-specific tests
pytest tests/unit/test_rag_pipeline.py -v
pytest tests/integration/test_rag_api.py -v

# With coverage
pytest --cov=src/news_rag
```

Tests use mocked external APIs and are safe to run without API keys.

## 11. Troubleshooting

### ChromaDB errors

```bash
# Reset the vector store
rm -rf .chroma_db
python scripts/run_app.py --reset-vector-store
```

### Missing API keys

The app will warn about missing keys but continue. Some features may not work:
- Without `GOOGLE_API_KEY`: No embeddings, summaries, or answers (Gemini)
- Without `TAVILY_API_KEY`: No news retrieval (GNews fallback may work)

### Port conflicts

```bash
# Use different ports
python scripts/run_app.py --backend-port 8080 --frontend-port 8502
```
