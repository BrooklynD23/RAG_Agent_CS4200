from typing import List

from ..models.news import Article
from ..tools.cache import get_cached, set_cached
from ..tools.tavily_tool import fetch_news_tavily
from ..tools.gnews_tool import fetch_news_gnews
from ..logging_config import get_logger


logger = get_logger("core.retrieval")


def retrieve_articles(topic: str, time_range: str = "7d", max_results: int = 10) -> List[Article]:
    """Retrieve news articles for a topic with caching and fallback.

    1. Check in-memory cache.
    2. Try Tavily tool.
    3. On failure, fall back to GNews.
    4. Store results in cache.
    """
    cached = get_cached(topic, time_range)
    if cached is not None:
        logger.info(
            "retrieve_articles_cache_hit",
            topic=topic,
            time_range=time_range,
            results=len(cached),
        )
        return cached

    try:
        articles = fetch_news_tavily(topic, max_results=max_results, time_range=time_range)
        backend = "tavily"
    except Exception:
        backend = "gnews"
        articles = fetch_news_gnews(topic, max_results=max_results, time_range=time_range)

    set_cached(topic, time_range, articles)
    logger.info(
        "retrieve_articles_fetched",
        topic=topic,
        time_range=time_range,
        backend=backend,
        results=len(articles),
    )
    return articles
