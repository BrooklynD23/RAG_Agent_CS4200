from typing import Literal, Optional, List, Dict, Any

from pydantic import BaseModel

from .news import Article, NewsSummary


class NewsState(BaseModel):
    query: str
    query_type: Literal["news", "general"]
    articles: List[Article] = []
    summary: Optional[NewsSummary] = None
    search_attempts: int = 0
    max_search_attempts: int = 3
    max_articles: int = 10
    time_range: str = "7d"
    verification_enabled: bool = True
    verification_result: Optional[Dict[str, Any]] = None
    status: Literal[
        "init",
        "searching",
        "summarizing",
        "verifying",
        "done",
        "failed",
    ] = "init"
    error: Optional[str] = None
