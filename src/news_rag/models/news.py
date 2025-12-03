from datetime import datetime
from typing import Optional, List, Dict

from pydantic import BaseModel


class Article(BaseModel):
    id: str
    title: str
    url: str
    source: str
    published_at: Optional[datetime] = None
    content: str
    score: Optional[float] = None


class SummarySentence(BaseModel):
    text: str
    source_ids: List[str]


class NewsSummary(BaseModel):
    topic: str
    summary_text: str
    sentences: List[SummarySentence]
    sources: List[Article]
    meta: Dict[str, object] = {}
