from src.news_rag.tools.tavily_tool import fetch_news_tavily
from src.news_rag.tools import tavily_tool, gnews_tool
from src.news_rag.tools.gnews_tool import fetch_news_gnews
import pytest
import types


def test_fetch_news_tavily_requires_api_key(monkeypatch) -> None:
    from src.news_rag import config

    monkeypatch.setattr(config.settings, "tavily_api_key", None)
    with pytest.raises(RuntimeError):
        fetch_news_tavily("test topic")


def test_fetch_news_tavily_maps_results(monkeypatch) -> None:
    class DummyClient:
        def search(self, *args, **kwargs):
            return {
                "results": [
                    {
                        "title": "Example",
                        "url": "https://example.com/article",
                        "content": "Body",
                        "score": 0.9,
                    }
                ]
            }

    # Ensure we do not use a real Tavily client or API key.
    monkeypatch.setattr(tavily_tool, "_client", None)
    monkeypatch.setattr(tavily_tool, "_get_client", lambda: DummyClient())

    articles = fetch_news_tavily("test topic", max_results=1)
    assert len(articles) == 1
    article = articles[0]
    assert article.title == "Example"
    assert article.url == "https://example.com/article"
    assert article.source == "example.com"
    assert article.content == "Body"
    assert article.score == 0.9


def test_fetch_news_gnews_uses_mocked_httpx(monkeypatch) -> None:
    from src.news_rag import config

    # Ensure we have an API key so the function does not early-exit.
    monkeypatch.setattr(config.settings, "gnews_api_key", "test-key")

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "articles": [
                    {
                        "id": "a1",
                        "title": "GNews Example",
                        "url": "https://news.example.com/item",
                        "source": {"name": "Example News"},
                        "publishedAt": "2024-01-01T00:00:00Z",
                        "content": "Story body",
                    }
                ]
            }

    def fake_get(url, params=None, timeout=None):
        assert "gnews.io" in url
        assert params["q"] == "topic"
        return DummyResponse()

    # Replace the httpx module used inside gnews_tool with a simple namespace.
    monkeypatch.setattr(gnews_tool, "httpx", types.SimpleNamespace(get=fake_get))

    articles = fetch_news_gnews("topic", max_results=1)
    assert len(articles) == 1
    article = articles[0]
    assert article.title == "GNews Example"
    assert article.source == "Example News"
    assert article.url == "https://news.example.com/item"
    assert article.content == "Story body"
