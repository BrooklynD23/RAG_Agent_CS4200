from fastapi.testclient import TestClient

from src.news_rag.api import server
from src.news_rag.api.server import app
from src.news_rag.models.news import Article, NewsSummary, SummarySentence
from src.news_rag.models.state import NewsState


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body.get("status") == "ok"


def test_summarize_endpoint_success_with_stubs(monkeypatch) -> None:
    def fake_retrieve_articles(topic: str, time_range: str = "7d", max_results: int = 10):
        return [
            Article(
                id="1",
                title="Title",
                url="https://example.com",
                source="example.com",
                content="Body",
            )
        ]

    def fake_summarize_articles(topic: str, articles):
        return NewsSummary(
            topic=topic,
            summary_text="Stub summary",
            sentences=[SummarySentence(text="Stub", source_ids=["1"])],
            sources=articles,
            meta={"stub": True},
        )

    def fake_verify_summary(summary, articles):
        return {"overall_verdict": "supported"}

    monkeypatch.setattr(server, "retrieve_articles", fake_retrieve_articles)
    monkeypatch.setattr(server, "summarize_articles", fake_summarize_articles)
    monkeypatch.setattr(server, "verify_summary", fake_verify_summary)

    response = client.post(
        "/summarize",
        json={
            "query": "topic",
            "time_range": "7d",
            "verification": True,
            "max_articles": 5,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["summary_text"] == "Stub summary"
    assert body["meta"]["verification"] is True
    assert body["meta"]["verification_result"]["overall_verdict"] == "supported"


def test_summarize_endpoint_handles_runtime_error(monkeypatch) -> None:
    def fake_retrieve_articles(topic: str, time_range: str = "7d", max_results: int = 10):
        return [
            Article(
                id="1",
                title="Title",
                url="https://example.com",
                source="example.com",
                content="Body",
            )
        ]

    def fake_summarize_articles(topic: str, articles):
        raise RuntimeError("missing OPENAI_API_KEY")

    monkeypatch.setattr(server, "retrieve_articles", fake_retrieve_articles)
    monkeypatch.setattr(server, "summarize_articles", fake_summarize_articles)

    response = client.post(
        "/summarize",
        json={
            "query": "topic",
            "time_range": "7d",
            "verification": True,
            "max_articles": 5,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["error"] == "missing OPENAI_API_KEY"
    assert len(body["sources"]) == 1


def test_debug_run_graph_endpoint_uses_stubbed_agent(monkeypatch) -> None:
    def fake_run_news_agent(query: str, time_range: str, verification: bool, max_articles: int, max_search_attempts: int):
        return NewsState(
            query=query,
            query_type="news",
            articles=[],
            summary=None,
            search_attempts=1,
            max_search_attempts=max_search_attempts,
            max_articles=max_articles,
            time_range=time_range,
            verification_enabled=verification,
            verification_result=None,
            status="done",
            error=None,
        )

    monkeypatch.setattr(server, "run_news_agent", fake_run_news_agent)

    response = client.post(
        "/debug/run-graph",
        json={
            "query": "topic",
            "time_range": "7d",
            "verification": False,
            "max_articles": 5,
            "max_search_attempts": 2,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "topic"
    assert body["status"] == "done"
