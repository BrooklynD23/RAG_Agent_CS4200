from fastapi import FastAPI
from pydantic import BaseModel

from ..core.router import classify_query
from ..core.retrieval import retrieve_articles
from ..core.summarization import summarize_articles
from ..core.verification import verify_summary
from ..core.graph import run_news_agent
from ..logging_config import get_logger


app = FastAPI()
logger = get_logger("api.server")


class SummarizeRequest(BaseModel):
    query: str
    time_range: str = "7d"
    verification: bool = True
    max_articles: int = 10


class DebugRunGraphRequest(BaseModel):
    query: str
    time_range: str = "7d"
    verification: bool = True
    max_articles: int = 10
    max_search_attempts: int = 3


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/summarize")
def summarize(req: SummarizeRequest) -> dict:
    logger.info(
        "summarize_request",
        query=req.query,
        time_range=req.time_range,
        verification=req.verification,
        max_articles=req.max_articles,
    )
    query_type = classify_query(req.query)
    articles = retrieve_articles(
        req.query,
        time_range=req.time_range,
        max_results=req.max_articles,
    )

    try:
        summary = summarize_articles(req.query, articles)
    except RuntimeError as exc:
        logger.warning("summarize_error", error=str(exc))
        return {
            "topic": req.query,
            "summary_text": "",
            "sentences": [],
            "sources": [a.dict() for a in articles],
            "meta": {
                "query_type": query_type,
                "time_range": req.time_range,
                "verification": req.verification,
                "error": str(exc),
            },
        }

    verification_result = None
    if req.verification:
        try:
            verification_result = verify_summary(summary, articles)
        except RuntimeError as exc:
            logger.warning("verification_error", error=str(exc))
            verification_result = {"error": str(exc)}

    response = summary.dict()
    meta = dict(response.get("meta") or {})
    meta.update(
        {
            "query_type": query_type,
            "time_range": req.time_range,
            "verification": req.verification,
            "verification_result": verification_result,
        }
    )
    response["meta"] = meta
    logger.info(
        "summarize_response",
        query=req.query,
        time_range=req.time_range,
        verification=req.verification,
        articles_count=len(articles),
    )
    return response


@app.post("/debug/run-graph")
def debug_run_graph(req: DebugRunGraphRequest) -> dict:
    """Execute the LangGraph agent and return the final NewsState.

    This endpoint is primarily intended for development and debugging.
    """

    logger.info(
        "debug_run_graph_request",
        query=req.query,
        time_range=req.time_range,
        verification=req.verification,
        max_articles=req.max_articles,
        max_search_attempts=req.max_search_attempts,
    )
    state = run_news_agent(
        query=req.query,
        time_range=req.time_range,
        verification=req.verification,
        max_articles=req.max_articles,
        max_search_attempts=req.max_search_attempts,
    )
    logger.info(
        "debug_run_graph_response",
        query=req.query,
        status=state.status,
        search_attempts=state.search_attempts,
    )
    return state.dict()
