"""RAG Agent Graph using LangGraph.

This module implements the full RAG pipeline as a LangGraph state machine,
handling both initial queries and follow-up questions with intelligent
routing and web search fallback.

Graph Structure:
    ┌─────────────────┐
    │   Entry Point   │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Classify Query │ ──► Determines initial vs follow-up
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌─────────┐    ┌──────────────┐
│ Initial │    │   Follow-up  │
│  Query  │    │    Query     │
└────┬────┘    └──────┬───────┘
     │                │
     ▼                ▼
┌─────────┐    ┌──────────────┐
│  Fetch  │    │   Retrieve   │
│  News   │    │   Chunks     │
└────┬────┘    └──────┬───────┘
     │                │
     ▼                ▼
┌─────────┐    ┌──────────────┐
│ Ingest  │    │    Check     │
│Articles │    │ Sufficiency  │
└────┬────┘    └──────┬───────┘
     │                │
     ▼         ┌──────┴──────┐
┌─────────┐    │             │
│Generate │    ▼             ▼
│ Summary │  Sufficient   Insufficient
└────┬────┘    │             │
     │         ▼             ▼
     │    ┌─────────┐  ┌──────────┐
     │    │Generate │  │Web Search│
     │    │ Answer  │  │ Fallback │
     │    └────┬────┘  └────┬─────┘
     │         │            │
     │         │            ▼
     │         │       ┌─────────┐
     │         │       │ Ingest  │
     │         │       │  New    │
     │         │       └────┬────┘
     │         │            │
     │         │            ▼
     │         │       ┌─────────┐
     │         │       │Generate │
     │         │       │ Answer  │
     │         │       └────┬────┘
     │         │            │
     └─────────┴────────────┘
               │
               ▼
          ┌─────────┐
          │   END   │
          └─────────┘
"""

from typing import Any, Dict, Literal, Optional

from langgraph.graph import StateGraph, END

from ..logging_config import get_logger
from ..models.news import Article, NewsSummary, SummarySentence
from ..models.rag_state import (
    AgentResponse,
    RAGState,
    RetrievedChunk,
    SourceReference,
    generate_id,
)
from .article_ingestor import ingest_articles
from .answer_generator import (
    generate_answer,
    generate_summary_answer,
    map_sources_used_to_references,
)
from .retrieval import retrieve_articles
from .sufficiency_checker import check_sufficiency
from .vector_retriever import (
    retrieve_relevant_chunks,
    chunks_to_source_references,
    format_chunks_for_context,
)

logger = get_logger("core.rag_graph")


# ============================================================================
# Graph Node Functions
# ============================================================================


def classify_message(state: RAGState) -> RAGState:
    """Classify the incoming message as initial query or follow-up.

    For now, this is based on whether we have articles already ingested
    for this conversation. A more sophisticated approach could use an LLM
    to classify based on message content.
    """
    # Check if this conversation already has ingested articles
    from .vector_store import get_chunks_by_conversation

    existing_chunks = get_chunks_by_conversation(state.conversation_id)

    if existing_chunks:
        message_type = "followup"
    else:
        message_type = "initial"

    logger.info(
        "message_classified",
        conversation_id=state.conversation_id,
        message_type=message_type,
        existing_chunks=len(existing_chunks),
    )

    return state.model_copy(
        update={
            "message_type": message_type,
            "status": "fetching_news" if message_type == "initial" else "retrieving",
        }
    )


def fetch_news(state: RAGState) -> RAGState:
    """Fetch news articles for the initial query."""
    logger.info(
        "fetching_news",
        query=state.query,
        time_range=state.time_range,
        max_articles=state.max_articles,
    )

    try:
        articles = retrieve_articles(
            topic=state.query,
            time_range=state.time_range,
            max_results=state.max_articles,
        )
    except Exception as exc:
        logger.error("fetch_news_failed", error=str(exc))
        return state.model_copy(
            update={
                "status": "failed",
                "error": f"Failed to fetch news: {str(exc)}",
            }
        )

    if not articles:
        logger.warning("no_articles_found", query=state.query)

    return state.model_copy(
        update={
            "articles": articles,
            "status": "ingesting",
        }
    )


def ingest_fetched_articles(state: RAGState) -> RAGState:
    """Ingest fetched articles into the vector store."""
    if not state.articles:
        return state.model_copy(
            update={
                "status": "generating_summary",
            }
        )

    logger.info(
        "ingesting_articles",
        conversation_id=state.conversation_id,
        articles=len(state.articles),
    )

    try:
        articles_count, chunks_count = ingest_articles(
            articles=state.articles,
            conversation_id=state.conversation_id,
        )
    except Exception as exc:
        logger.error("ingest_failed", error=str(exc))
        # Continue anyway - we can still generate a summary from articles
        chunks_count = 0

    return state.model_copy(
        update={
            "status": "generating_summary",
            "debug_info": {
                **state.debug_info,
                "articles_ingested": len(state.articles),
                "chunks_stored": chunks_count,
            },
        }
    )


def generate_summary(state: RAGState) -> RAGState:
    """Generate a summary for the initial query."""
    logger.info(
        "generating_summary",
        query=state.query,
        articles=len(state.articles),
    )

    # Retrieve chunks for summary generation
    chunks = retrieve_relevant_chunks(
        query=state.query,
        conversation_id=state.conversation_id,
        max_chunks=state.max_chunks,
    )

    if not chunks and not state.articles:
        return state.model_copy(
            update={
                "answer_text": "No relevant news articles were found for this topic.",
                "answer_type": "summary",
                "sources_used": [],
                "status": "done",
            }
        )

    # Generate summary from chunks
    result = generate_summary_answer(state.query, chunks)

    # Map sources
    sources_used = map_sources_used_to_references(
        result.get("sources_used", []),
        chunks,
    )

    # If no sources mapped, use all unique sources from chunks
    if not sources_used:
        sources_used = chunks_to_source_references(chunks)

    # Create NewsSummary for compatibility with existing code
    summary = NewsSummary(
        topic=state.query,
        summary_text=result.get("answer", ""),
        sentences=[],  # Could parse bullet points into sentences
        sources=state.articles,
        meta={
            "confidence": result.get("confidence"),
            "missing_info": result.get("missing_info"),
        },
    )

    return state.model_copy(
        update={
            "summary": summary,
            "answer_text": result.get("answer", ""),
            "answer_type": "summary",
            "sources_used": sources_used,
            "status": "done",
            "debug_info": {
                **state.debug_info,
                "chunks_used": len(chunks),
                "confidence": result.get("confidence"),
            },
        }
    )


def retrieve_chunks(state: RAGState) -> RAGState:
    """Retrieve relevant chunks for a follow-up question."""
    logger.info(
        "retrieving_chunks",
        query=state.query,
        conversation_id=state.conversation_id,
    )

    chunks = retrieve_relevant_chunks(
        query=state.query,
        conversation_id=state.conversation_id,
        max_chunks=state.max_chunks,
        similarity_threshold=state.similarity_threshold,
    )

    return state.model_copy(
        update={
            "retrieved_chunks": chunks,
            "status": "checking_sufficiency",
            "debug_info": {
                **state.debug_info,
                "chunks_retrieved": len(chunks),
                "top_similarity": chunks[0].similarity_score if chunks else 0,
            },
        }
    )


def check_retrieval_sufficiency(state: RAGState) -> RAGState:
    """Check if retrieved chunks are sufficient to answer the question."""
    is_sufficient, reason = check_sufficiency(
        query=state.query,
        chunks=state.retrieved_chunks,
        use_llm=False,  # Use heuristic for speed
    )

    logger.info(
        "sufficiency_checked",
        sufficient=is_sufficient,
        reason=reason,
        chunks=len(state.retrieved_chunks),
    )

    return state.model_copy(
        update={
            "retrieval_sufficient": is_sufficient,
            "sufficiency_reason": reason,
            "status": "generating_answer" if is_sufficient else "web_searching",
        }
    )


def web_search_fallback(state: RAGState) -> RAGState:
    """Fetch additional articles via web search when stored sources are insufficient."""
    logger.info(
        "web_search_fallback",
        query=state.query,
        reason=state.sufficiency_reason,
    )

    try:
        new_articles = retrieve_articles(
            topic=state.query,
            time_range=state.time_range,
            max_results=state.max_articles,
        )
    except Exception as exc:
        logger.error("web_search_failed", error=str(exc))
        # Continue with what we have
        return state.model_copy(
            update={
                "web_search_triggered": True,
                "status": "generating_answer",
            }
        )

    return state.model_copy(
        update={
            "new_articles": new_articles,
            "web_search_triggered": True,
            "status": "ingesting",
        }
    )


def ingest_new_articles(state: RAGState) -> RAGState:
    """Ingest newly fetched articles from web search."""
    if not state.new_articles:
        return state.model_copy(update={"status": "generating_answer"})

    logger.info(
        "ingesting_new_articles",
        conversation_id=state.conversation_id,
        articles=len(state.new_articles),
    )

    try:
        _, chunks_count = ingest_articles(
            articles=state.new_articles,
            conversation_id=state.conversation_id,
        )
    except Exception as exc:
        logger.error("ingest_new_failed", error=str(exc))
        chunks_count = 0

    # Re-retrieve chunks after ingestion
    updated_chunks = retrieve_relevant_chunks(
        query=state.query,
        conversation_id=state.conversation_id,
        max_chunks=state.max_chunks,
    )

    return state.model_copy(
        update={
            "retrieved_chunks": updated_chunks,
            "status": "generating_answer",
            "debug_info": {
                **state.debug_info,
                "new_articles_ingested": len(state.new_articles),
                "new_chunks_stored": chunks_count,
            },
        }
    )


def generate_followup_answer(state: RAGState) -> RAGState:
    """Generate an answer for a follow-up question."""
    logger.info(
        "generating_followup_answer",
        query=state.query,
        chunks=len(state.retrieved_chunks),
        web_augmented=state.web_search_triggered,
    )

    result = generate_answer(
        query=state.query,
        chunks=state.retrieved_chunks,
        is_followup=True,
    )

    sources_used = map_sources_used_to_references(
        result.get("sources_used", []),
        state.retrieved_chunks,
    )

    if not sources_used:
        sources_used = chunks_to_source_references(state.retrieved_chunks)

    answer_type = "web_augmented_answer" if state.web_search_triggered else "followup_answer"

    return state.model_copy(
        update={
            "answer_text": result.get("answer", ""),
            "answer_type": answer_type,
            "sources_used": sources_used,
            "status": "done",
            "debug_info": {
                **state.debug_info,
                "confidence": result.get("confidence"),
                "missing_info": result.get("missing_info"),
            },
        }
    )


def handle_error(state: RAGState) -> RAGState:
    """Handle errors in the pipeline."""
    logger.error(
        "pipeline_error",
        error=state.error,
        status=state.status,
    )
    return state


# ============================================================================
# Routing Functions
# ============================================================================


def route_by_message_type(state: RAGState) -> Literal["initial", "followup"]:
    """Route based on whether this is an initial query or follow-up."""
    return state.message_type


def route_by_sufficiency(state: RAGState) -> Literal["sufficient", "insufficient"]:
    """Route based on whether retrieved chunks are sufficient."""
    return "sufficient" if state.retrieval_sufficient else "insufficient"


def route_after_web_ingest(state: RAGState) -> Literal["generate", "error"]:
    """Route after ingesting new articles from web search."""
    if state.error:
        return "error"
    return "generate"


# ============================================================================
# Graph Builder
# ============================================================================


def build_rag_graph() -> StateGraph:
    """Build and compile the RAG agent graph."""

    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("classify_message", classify_message)
    graph.add_node("fetch_news", fetch_news)
    graph.add_node("ingest_articles", ingest_fetched_articles)
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("retrieve_chunks", retrieve_chunks)
    graph.add_node("check_sufficiency", check_retrieval_sufficiency)
    graph.add_node("web_search", web_search_fallback)
    graph.add_node("ingest_new_articles", ingest_new_articles)
    graph.add_node("generate_answer", generate_followup_answer)
    graph.add_node("handle_error", handle_error)

    # Set entry point
    graph.set_entry_point("classify_message")

    # Add conditional edge after classification
    graph.add_conditional_edges(
        "classify_message",
        route_by_message_type,
        {
            "initial": "fetch_news",
            "followup": "retrieve_chunks",
        },
    )

    # Initial query flow
    graph.add_edge("fetch_news", "ingest_articles")
    graph.add_edge("ingest_articles", "generate_summary")
    graph.add_edge("generate_summary", END)

    # Follow-up query flow
    graph.add_edge("retrieve_chunks", "check_sufficiency")

    graph.add_conditional_edges(
        "check_sufficiency",
        route_by_sufficiency,
        {
            "sufficient": "generate_answer",
            "insufficient": "web_search",
        },
    )

    # Web search fallback flow
    graph.add_edge("web_search", "ingest_new_articles")
    graph.add_edge("ingest_new_articles", "generate_answer")

    # Answer generation to end
    graph.add_edge("generate_answer", END)

    # Error handling
    graph.add_edge("handle_error", END)

    return graph.compile()


# ============================================================================
# Public API
# ============================================================================


def run_news_query(
    user_id: Optional[str],
    conversation_id: Optional[str],
    message: str,
    time_range: str = "7d",
    max_articles: int = 10,
    max_chunks: int = 10,
    include_debug: bool = False,
) -> AgentResponse:
    """Main entry point for the RAG news agent.

    Args:
        user_id: Optional user identifier.
        conversation_id: Optional conversation ID. If None, a new conversation is created.
        message: The user's query or follow-up question.
        time_range: Time range for news search (e.g., "24h", "7d", "30d").
        max_articles: Maximum articles to fetch.
        max_chunks: Maximum chunks to retrieve.
        include_debug: Whether to include debug info in response.

    Returns:
        AgentResponse with answer, sources, and optional debug info.
    """
    # Create or use existing conversation ID
    conv_id = conversation_id or generate_id()

    # Build initial state
    initial_state = RAGState(
        query=message,
        conversation_id=conv_id,
        user_id=user_id,
        time_range=time_range,
        max_articles=max_articles,
        max_chunks=max_chunks,
    )

    logger.info(
        "running_news_query",
        user_id=user_id,
        conversation_id=conv_id,
        query_preview=message[:50],
    )

    # Build and run the graph
    app = build_rag_graph()
    result = app.invoke(initial_state.model_dump())

    # Convert result back to RAGState
    final_state = RAGState(**result)

    # Build response
    return AgentResponse.from_state(final_state, include_debug=include_debug)


def get_conversation_sources(conversation_id: str) -> list[SourceReference]:
    """Get all sources for a conversation.

    Args:
        conversation_id: The conversation ID.

    Returns:
        List of unique SourceReference objects.
    """
    from .vector_store import get_chunks_by_conversation

    chunks = get_chunks_by_conversation(conversation_id)
    return chunks_to_source_references(chunks)


def clear_conversation(conversation_id: str) -> int:
    """Clear all stored data for a conversation.

    Args:
        conversation_id: The conversation ID to clear.

    Returns:
        Number of chunks deleted.
    """
    from .vector_store import delete_conversation_chunks

    return delete_conversation_chunks(conversation_id)
