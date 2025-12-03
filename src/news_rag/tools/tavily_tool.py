from typing import List
from urllib.parse import urlparse

from tavily import TavilyClient

from ..config import settings
from ..models.news import Article


_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    """Return a shared TavilyClient instance, ensuring the API key is set."""

    global _client
    if _client is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured in the environment.")
        _client = TavilyClient(api_key=settings.tavily_api_key)
    return _client


def _source_from_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or ""


def fetch_news_tavily(topic: str, max_results: int = 10, time_range: str = "7d") -> List[Article]:
    """Fetch news articles for a topic using the Tavily Search API.

    Uses the official Tavily Python client. Results are mapped into Article
    models; only a subset of Tavily fields is used here.
    """

    client = _get_client()

    # We rely on Tavily's own ranking and news topic. Time range is kept as
    # a parameter for future use but not strictly required.
    response = client.search(
        topic,
        topic="news",
        search_depth="basic",
        max_results=max_results,
        include_answer=False,
        include_raw_content=True,
    )

    results = response.get("results", []) or []
    articles: List[Article] = []
    for idx, item in enumerate(results):
        url = item.get("url") or ""
        content = item.get("content") or item.get("raw_content") or ""
        if not content:
            continue

        score = item.get("score")
        articles.append(
            Article(
                id=str(idx + 1),
                title=item.get("title") or "",
                url=url,
                source=_source_from_url(url),
                published_at=None,
                content=content,
                score=float(score) if score is not None else None,
            )
        )

    return articles
