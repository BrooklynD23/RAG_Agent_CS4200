"""Article ingestion: chunking, embedding, and storage.

This module handles the transformation of raw Article objects into
chunked, embedded representations stored in the vector database.
"""

import re
from typing import List, Tuple
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings
from ..logging_config import get_logger
from ..models.news import Article
from ..models.rag_state import ArticleChunk
from .vector_store import add_chunks

logger = get_logger("core.article_ingestor")


def _clean_text(text: str) -> str:
    """Clean article text for better chunking.

    - Normalize whitespace
    - Remove excessive newlines
    - Strip leading/trailing whitespace
    """
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse multiple newlines into double newlines (paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip
    return text.strip()


def _create_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create a text splitter with configured chunk size and overlap."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_article(article: Article, conversation_id: str) -> List[ArticleChunk]:
    """Split an article into chunks with metadata.

    Args:
        article: The Article to chunk.
        conversation_id: The conversation/topic ID to tag chunks with.

    Returns:
        List of ArticleChunk objects.
    """
    cleaned_content = _clean_text(article.content)

    if not cleaned_content:
        logger.warning(
            "empty_article_content",
            article_id=article.id,
            title=article.title,
        )
        return []

    splitter = _create_text_splitter()
    text_chunks = splitter.split_text(cleaned_content)

    chunks: List[ArticleChunk] = []
    for idx, chunk_text in enumerate(text_chunks):
        chunk_id = f"{article.id}_{conversation_id}_{idx}_{uuid4().hex[:6]}"
        chunks.append(
            ArticleChunk(
                chunk_id=chunk_id,
                article_id=article.id,
                conversation_id=conversation_id,
                content=chunk_text,
                chunk_index=idx,
                url=article.url,
                title=article.title,
                source=article.source,
                published_at=article.published_at,
            )
        )

    logger.info(
        "article_chunked",
        article_id=article.id,
        title=article.title[:50] if article.title else "",
        chunks=len(chunks),
    )

    return chunks


def ingest_articles(
    articles: List[Article],
    conversation_id: str,
) -> Tuple[int, int]:
    """Ingest a list of articles into the vector store.

    This function:
    1. Cleans each article's content
    2. Splits into chunks
    3. Embeds and stores in the vector database

    Args:
        articles: List of Article objects to ingest.
        conversation_id: The conversation/topic ID to tag all chunks with.

    Returns:
        Tuple of (articles_processed, chunks_stored).
    """
    if not articles:
        return 0, 0

    all_chunks: List[ArticleChunk] = []

    for article in articles:
        chunks = chunk_article(article, conversation_id)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning(
            "no_chunks_generated",
            articles=len(articles),
            conversation_id=conversation_id,
        )
        return len(articles), 0

    # Store chunks in vector database
    try:
        stored_count = add_chunks(all_chunks)
    except RuntimeError as exc:
        logger.error(
            "ingest_failed",
            error=str(exc),
            articles=len(articles),
            chunks=len(all_chunks),
        )
        raise

    logger.info(
        "articles_ingested",
        articles=len(articles),
        chunks=stored_count,
        conversation_id=conversation_id,
    )

    return len(articles), stored_count


def ingest_single_article(article: Article, conversation_id: str) -> int:
    """Convenience function to ingest a single article.

    Args:
        article: The Article to ingest.
        conversation_id: The conversation/topic ID.

    Returns:
        Number of chunks stored.
    """
    _, chunks_stored = ingest_articles([article], conversation_id)
    return chunks_stored


def get_article_ids_from_chunks(chunks: List[ArticleChunk]) -> List[str]:
    """Extract unique article IDs from a list of chunks."""
    seen = set()
    article_ids = []
    for chunk in chunks:
        if chunk.article_id not in seen:
            seen.add(chunk.article_id)
            article_ids.append(chunk.article_id)
    return article_ids
