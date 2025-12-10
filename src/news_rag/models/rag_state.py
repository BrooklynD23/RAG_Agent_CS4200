"""RAG-specific state models for conversation and topic tracking.

These models extend the base news models to support:
- Conversation/session tracking
- Topic-based article grouping
- Chunk-level metadata for retrieval
- Source citation mapping
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from .news import Article, NewsSummary


def generate_id() -> str:
    """Generate a unique ID for conversations/topics."""
    return uuid4().hex[:12]


class ArticleChunk(BaseModel):
    """A chunk of an article with metadata for vector storage."""

    chunk_id: str
    article_id: str
    conversation_id: str
    content: str
    chunk_index: int

    # Metadata from parent article
    url: str
    title: str
    source: str
    published_at: Optional[datetime] = None

    # Embedding info (populated after embedding)
    embedding: Optional[List[float]] = None

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dict for vector store."""
        data: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "article_id": self.article_id,
            "conversation_id": self.conversation_id,
            "chunk_index": self.chunk_index,
            "url": self.url,
            "title": self.title,
            "source": self.source,
            "published_at": self.published_at.isoformat() if self.published_at else None,
        }

        # Chroma's metadata schema does not accept None values; drop them
        return {k: v for k, v in data.items() if v is not None}


class RetrievedChunk(BaseModel):
    """A chunk retrieved from the vector store with similarity score."""

    chunk_id: str
    article_id: str
    conversation_id: str
    content: str
    chunk_index: int
    url: str
    title: str
    source: str
    published_at: Optional[datetime] = None
    similarity_score: float = 0.0


class SourceReference(BaseModel):
    """A source reference for citation in answers."""

    article_id: str
    url: str
    title: str
    source: str
    published_at: Optional[datetime] = None

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> "SourceReference":
        return cls(
            article_id=chunk.article_id,
            url=chunk.url,
            title=chunk.title,
            source=chunk.source,
            published_at=chunk.published_at,
        )


class ConversationContext(BaseModel):
    """Tracks the context of a conversation/session."""

    conversation_id: str = Field(default_factory=generate_id)
    user_id: Optional[str] = None
    initial_query: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Articles ingested for this conversation
    article_ids: List[str] = []

    # Summary generated for initial query
    summary: Optional[NewsSummary] = None


class RAGState(BaseModel):
    """State for the RAG agent graph.

    This extends NewsState with RAG-specific fields for:
    - Conversation tracking
    - Retrieved chunks
    - Sufficiency assessment
    - Answer generation
    """

    # Query info
    query: str
    message_type: Literal["initial", "followup"] = "initial"

    # Conversation tracking
    conversation_id: str = Field(default_factory=generate_id)
    user_id: Optional[str] = None

    # Search/retrieval settings
    time_range: str = "7d"
    max_articles: int = 10
    max_chunks: int = 10
    similarity_threshold: float = 0.7

    # Articles (for initial query)
    articles: List[Article] = []

    # Retrieved chunks (for follow-up or grounding)
    retrieved_chunks: List[RetrievedChunk] = []

    # Sufficiency assessment
    retrieval_sufficient: bool = False
    sufficiency_reason: Optional[str] = None

    # Web search fallback
    web_search_triggered: bool = False
    new_articles: List[Article] = []

    # Generated outputs
    summary: Optional[NewsSummary] = None
    answer_text: Optional[str] = None
    answer_type: Literal["summary", "followup_answer", "web_augmented_answer"] = "summary"

    # Sources used in answer
    sources_used: List[SourceReference] = []

    # Status tracking
    status: Literal[
        "init",
        "fetching_news",
        "ingesting",
        "retrieving",
        "checking_sufficiency",
        "web_searching",
        "generating_summary",
        "generating_answer",
        "done",
        "failed",
    ] = "init"
    error: Optional[str] = None

    # Debug info
    debug_info: Dict[str, Any] = {}


class AgentResponse(BaseModel):
    """Response from the RAG agent to be returned via API."""

    answer_text: str
    answer_type: Literal["summary", "followup_answer", "web_augmented_answer"]
    sources: List[SourceReference]
    conversation_id: str

    # Optional debug info
    debug: Optional[Dict[str, Any]] = None

    @classmethod
    def from_state(cls, state: RAGState, include_debug: bool = False) -> "AgentResponse":
        answer = state.answer_text or ""
        if state.summary and not answer:
            answer = state.summary.summary_text

        return cls(
            answer_text=answer,
            answer_type=state.answer_type,
            sources=state.sources_used,
            conversation_id=state.conversation_id,
            debug=state.debug_info if include_debug else None,
        )
