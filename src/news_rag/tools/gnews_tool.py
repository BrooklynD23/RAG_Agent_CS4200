from datetime import datetime
from typing import List
from urllib.parse import urlparse

import httpx

from ..config import settings
from ..models.news import Article


_GNEWS_SEARCH_URL = "https://gnews.io/api/v4/search"


def _parse_published_at(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def fetch_news_gnews(topic: str, max_results: int = 10, time_range: str = "7d") -> List[Article]:
    """Fallback news retrieval using the GNews REST API.

    If `GNEWS_API_KEY` is not configured, this returns an empty list rather
    than raising, so callers can degrade gracefully.
    """

    if not settings.gnews_api_key:
        return []

    params = {
        "q": topic,
        "lang": "en",
        "max": max_results,
        "apikey": settings.gnews_api_key,
    }

    try:
        response = httpx.get(_GNEWS_SEARCH_URL, params=params, timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return []

    data = response.json()
    raw_articles = data.get("articles") or data.get("data") or []

    articles: List[Article] = []
    for idx, item in enumerate(raw_articles):
        url = item.get("url") or ""
        parsed = urlparse(url)
        source_obj = item.get("source") or {}
        source_name = source_obj.get("name") or parsed.netloc
        published_at = _parse_published_at(item.get("publishedAt"))
        content = item.get("content") or item.get("description") or ""
        if not content:
            continue

        articles.append(
            Article(
                id=item.get("id") or str(idx + 1),
                title=item.get("title") or "",
                url=url,
                source=source_name,
                published_at=published_at,
                content=content,
                score=None,
            )
        )

    return articles
