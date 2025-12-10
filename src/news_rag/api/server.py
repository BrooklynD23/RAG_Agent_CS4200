from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..core.router import classify_query
from ..core.retrieval import retrieve_articles
from ..core.summarization import summarize_articles
from ..core.verification import verify_summary
from ..core.graph import run_news_agent
from ..core.rag_graph import (
    run_news_query,
    get_conversation_sources,
    clear_conversation,
)
from ..core.vector_store import get_collection_stats
from ..models.rag_state import AgentResponse, SourceReference
from ..logging_config import get_logger


app = FastAPI(
    title="News RAG Agent API",
    description="RAG-based news summarization and Q&A agent",
    version="2.0.0",
)
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


# ============================================================================
# RAG Agent Request/Response Models
# ============================================================================


class RAGQueryRequest(BaseModel):
    """Request model for RAG agent queries."""

    message: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    time_range: str = "7d"
    max_articles: int = 10
    max_chunks: int = 10
    include_debug: bool = False


class RAGSourceResponse(BaseModel):
    """Source reference in API response."""

    article_id: str
    url: str
    title: str
    source: str
    published_at: Optional[str] = None


class RAGQueryResponse(BaseModel):
    """Response model for RAG agent queries."""

    answer_text: str
    answer_type: str  # "summary", "followup_answer", "web_augmented_answer"
    sources: List[RAGSourceResponse]
    conversation_id: str
    debug: Optional[dict] = None


# ============================================================================
# Health and Status Endpoints
# ============================================================================


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


# ============================================================================
# RAG Agent Endpoints (New)
# ============================================================================


@app.post("/rag/query", response_model=RAGQueryResponse)
def rag_query(req: RAGQueryRequest) -> RAGQueryResponse:
    """Execute a RAG query - handles both initial queries and follow-ups.

    This is the main endpoint for the RAG news agent. It:
    - For initial queries: fetches news, ingests into vector DB, returns summary
    - For follow-ups: retrieves from vector DB, optionally searches web if needed

    The conversation_id is used to track which articles belong to which conversation.
    Pass the same conversation_id for follow-up questions to retrieve from the
    same article corpus.
    """
    logger.info(
        "rag_query_request",
        message=req.message[:50],
        user_id=req.user_id,
        conversation_id=req.conversation_id,
        time_range=req.time_range,
    )

    try:
        response = run_news_query(
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            message=req.message,
            time_range=req.time_range,
            max_articles=req.max_articles,
            max_chunks=req.max_chunks,
            include_debug=req.include_debug,
        )
    except Exception as exc:
        logger.error("rag_query_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

    # Convert SourceReference to RAGSourceResponse
    sources = [
        RAGSourceResponse(
            article_id=s.article_id,
            url=s.url,
            title=s.title,
            source=s.source,
            published_at=s.published_at.isoformat() if s.published_at else None,
        )
        for s in response.sources
    ]

    logger.info(
        "rag_query_response",
        conversation_id=response.conversation_id,
        answer_type=response.answer_type,
        sources_count=len(sources),
    )

    return RAGQueryResponse(
        answer_text=response.answer_text,
        answer_type=response.answer_type,
        sources=sources,
        conversation_id=response.conversation_id,
        debug=response.debug,
    )


@app.get("/rag/conversation/{conversation_id}/sources")
def get_sources(conversation_id: str) -> dict:
    """Get all sources stored for a conversation.

    Returns the list of unique articles that have been ingested
    for this conversation.
    """
    try:
        sources = get_conversation_sources(conversation_id)
        return {
            "conversation_id": conversation_id,
            "sources": [
                {
                    "article_id": s.article_id,
                    "url": s.url,
                    "title": s.title,
                    "source": s.source,
                    "published_at": s.published_at.isoformat() if s.published_at else None,
                }
                for s in sources
            ],
            "count": len(sources),
        }
    except Exception as exc:
        logger.error("get_sources_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/rag/conversation/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict:
    """Delete all stored data for a conversation.

    This removes all article chunks from the vector store for the
    specified conversation.
    """
    try:
        deleted_count = clear_conversation(conversation_id)
        logger.info(
            "conversation_deleted",
            conversation_id=conversation_id,
            chunks_deleted=deleted_count,
        )
        return {
            "conversation_id": conversation_id,
            "chunks_deleted": deleted_count,
            "status": "deleted",
        }
    except Exception as exc:
        logger.error("delete_conversation_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/rag/stats")
def get_stats() -> dict:
    """Get statistics about the vector store.

    Returns information about the ChromaDB collection including
    total document count.
    """
    try:
        stats = get_collection_stats()
        return {
            "vector_store": stats,
            "status": "ok",
        }
    except Exception as exc:
        logger.error("get_stats_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
