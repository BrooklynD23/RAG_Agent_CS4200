from datetime import datetime, timedelta
from typing import Dict, Tuple, List

from ..models.news import Article


_CACHE: Dict[Tuple[str, str], Tuple[datetime, List[Article]]] = {}
_TTL = timedelta(minutes=30)


def get_cached(query: str, time_range: str) -> List[Article] | None:
    key = (query, time_range)
    entry = _CACHE.get(key)
    if not entry:
        return None
    ts, articles = entry
    if datetime.utcnow() - ts > _TTL:
        _CACHE.pop(key, None)
        return None
    return articles


def set_cached(query: str, time_range: str, articles: List[Article]) -> None:
    _CACHE[(query, time_range)] = (datetime.utcnow(), articles)
