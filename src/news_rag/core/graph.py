from typing import Literal

from langgraph.graph import StateGraph, END

from ..models.state import NewsState
from .router import classify_query
from .retrieval import retrieve_articles
from .summarization import summarize_articles
from .verification import verify_summary


def route_query(state: NewsState) -> NewsState:
    """Classify the query as news vs general and set initial status."""

    query_type = classify_query(state.query)
    return state.copy(update={"query_type": query_type, "status": "searching"})


def search_news(state: NewsState) -> NewsState:
    """Search for news articles using the retrieval layer."""

    articles = retrieve_articles(
        state.query,
        time_range=state.time_range,
        max_results=state.max_articles,
    )
    return state.copy(
        update={
            "articles": articles,
            "search_attempts": state.search_attempts + 1,
            "status": "searching",
        }
    )


def grade_results(state: NewsState) -> NewsState:
    """Lightweight grading step to decide whether to re-search or summarize.

    This does not call an LLM; it uses simple heuristics over the number
    of articles and attempts. The conditional edge function below decides
    the next node.
    """

    if not state.articles and state.search_attempts >= state.max_search_attempts:
        return state.copy(update={"status": "failed", "error": "no_articles"})
    return state


def _grade_decision(state: NewsState) -> Literal["search_more", "summarize", "fail"]:
    if not state.articles and state.search_attempts >= state.max_search_attempts:
        return "fail"
    if len(state.articles) < 3 and state.search_attempts < state.max_search_attempts:
        return "search_more"
    return "summarize"


def summarize_news(state: NewsState) -> NewsState:
    """Run the summarization chain over the collected articles."""

    summary = summarize_articles(state.query, state.articles)
    new_status = "verifying" if state.verification_enabled else "done"
    return state.copy(update={"summary": summary, "status": new_status})


def _summarize_decision(state: NewsState) -> Literal["verify", "end"]:
    return "verify" if state.verification_enabled else "end"


def verify_news(state: NewsState) -> NewsState:
    """Optional verification/critic step over the summary."""

    if state.summary is None:
        return state.copy(update={"status": "failed", "error": "no_summary"})

    try:
        verdict = verify_summary(state.summary, state.articles)
        return state.copy(
            update={
                "verification_result": verdict,
                "status": "done",
            }
        )
    except RuntimeError as exc:
        # Surface verification issues but still return a usable summary.
        return state.copy(update={"status": "done", "error": str(exc)})


def handle_error(state: NewsState) -> NewsState:
    """Terminal error node."""

    return state


def build_news_agent_graph():
    """Build and return the LangGraph app for the news agent."""

    graph = StateGraph(NewsState)

    graph.add_node("route_query", route_query)
    graph.add_node("search_news", search_news)
    graph.add_node("grade_results", grade_results)
    graph.add_node("summarize_news", summarize_news)
    graph.add_node("verify_news", verify_news)
    graph.add_node("handle_error", handle_error)

    graph.set_entry_point("route_query")
    graph.add_edge("route_query", "search_news")
    graph.add_edge("search_news", "grade_results")

    graph.add_conditional_edges(
        "grade_results",
        _grade_decision,
        {
            "search_more": "search_news",
            "summarize": "summarize_news",
            "fail": "handle_error",
        },
    )

    graph.add_conditional_edges(
        "summarize_news",
        _summarize_decision,
        {
            "verify": "verify_news",
            "end": END,
        },
    )

    graph.add_edge("verify_news", END)
    graph.add_edge("handle_error", END)

    return graph.compile()


def run_news_agent(
    query: str,
    time_range: str = "7d",
    verification: bool = True,
    max_articles: int = 10,
    max_search_attempts: int = 3,
) -> NewsState:
    """Helper to execute the LangGraph agent and return the final state.

    This is primarily used by the `/debug/run-graph` endpoint for
    inspection and debugging.
    """

    app = build_news_agent_graph()
    initial_state = NewsState(
        query=query,
        query_type="news",  # will be refined by route_query
        time_range=time_range,
        verification_enabled=verification,
        max_articles=max_articles,
        max_search_attempts=max_search_attempts,
    )

    # LangGraph apps operate on dict-like state; convert to/from NewsState.
    result = app.invoke(initial_state.dict())
    return NewsState(**result)
