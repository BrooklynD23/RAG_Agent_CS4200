"""Vector retrieval for RAG queries.

This module provides high-level retrieval functions that query the
vector store and return relevant chunks for answering user questions.
"""

from typing import Dict, List, Optional, Set

from ..logging_config import get_logger
from ..models.rag_state import RetrievedChunk, SourceReference
from .vector_store import query_chunks, get_chunks_by_conversation

logger = get_logger("core.vector_retriever")


def retrieve_relevant_chunks(
    query: str,
    conversation_id: Optional[str] = None,
    max_chunks: int = 10,
    similarity_threshold: float = 0.3,
) -> List[RetrievedChunk]:
    """Retrieve the most relevant chunks for a query.

    Args:
        query: The user's question or search query.
        conversation_id: Optional filter to search within a specific conversation.
        max_chunks: Maximum number of chunks to return.
        similarity_threshold: Minimum similarity score to include.

    Returns:
        List of RetrievedChunk objects sorted by relevance.
    """
    chunks = query_chunks(
        query=query,
        conversation_id=conversation_id,
        n_results=max_chunks,
        similarity_threshold=similarity_threshold,
    )

    logger.info(
        "chunks_retrieved",
        query_preview=query[:50],
        conversation_id=conversation_id,
        results=len(chunks),
        threshold=similarity_threshold,
    )

    return chunks


def retrieve_with_context_expansion(
    query: str,
    conversation_id: str,
    max_chunks: int = 10,
    similarity_threshold: float = 0.3,
    expand_context: bool = True,
) -> List[RetrievedChunk]:
    """Retrieve chunks with optional context expansion.

    If expand_context is True, for each retrieved chunk, also include
    adjacent chunks from the same article to provide more context.

    Args:
        query: The user's question.
        conversation_id: The conversation to search within.
        max_chunks: Maximum number of primary chunks to retrieve.
        similarity_threshold: Minimum similarity score.
        expand_context: Whether to include adjacent chunks.

    Returns:
        List of RetrievedChunk objects with expanded context.
    """
    # Get primary chunks
    primary_chunks = retrieve_relevant_chunks(
        query=query,
        conversation_id=conversation_id,
        max_chunks=max_chunks,
        similarity_threshold=similarity_threshold,
    )

    if not expand_context or not primary_chunks:
        return primary_chunks

    # Get all chunks for the conversation to find adjacent ones
    all_conversation_chunks = get_chunks_by_conversation(conversation_id)

    # Build index by article_id and chunk_index
    chunk_index: Dict[str, Dict[int, RetrievedChunk]] = {}
    for chunk in all_conversation_chunks:
        if chunk.article_id not in chunk_index:
            chunk_index[chunk.article_id] = {}
        chunk_index[chunk.article_id][chunk.chunk_index] = chunk

    # Expand context by including adjacent chunks
    expanded_chunks: List[RetrievedChunk] = []
    seen_chunk_ids: Set[str] = set()

    for chunk in primary_chunks:
        article_chunks = chunk_index.get(chunk.article_id, {})

        # Add previous chunk if exists
        prev_idx = chunk.chunk_index - 1
        if prev_idx in article_chunks:
            prev_chunk = article_chunks[prev_idx]
            if prev_chunk.chunk_id not in seen_chunk_ids:
                # Give it a slightly lower score than the primary chunk
                prev_chunk.similarity_score = chunk.similarity_score * 0.9
                expanded_chunks.append(prev_chunk)
                seen_chunk_ids.add(prev_chunk.chunk_id)

        # Add the primary chunk
        if chunk.chunk_id not in seen_chunk_ids:
            expanded_chunks.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)

        # Add next chunk if exists
        next_idx = chunk.chunk_index + 1
        if next_idx in article_chunks:
            next_chunk = article_chunks[next_idx]
            if next_chunk.chunk_id not in seen_chunk_ids:
                next_chunk.similarity_score = chunk.similarity_score * 0.9
                expanded_chunks.append(next_chunk)
                seen_chunk_ids.add(next_chunk.chunk_id)

    # Re-sort by similarity
    expanded_chunks.sort(key=lambda x: x.similarity_score, reverse=True)

    logger.info(
        "context_expanded",
        primary_chunks=len(primary_chunks),
        expanded_chunks=len(expanded_chunks),
    )

    return expanded_chunks


def chunks_to_source_references(chunks: List[RetrievedChunk]) -> List[SourceReference]:
    """Convert retrieved chunks to unique source references.

    Deduplicates by article_id to avoid listing the same source multiple times.

    Args:
        chunks: List of retrieved chunks.

    Returns:
        List of unique SourceReference objects.
    """
    seen_article_ids: Set[str] = set()
    sources: List[SourceReference] = []

    for chunk in chunks:
        if chunk.article_id not in seen_article_ids:
            seen_article_ids.add(chunk.article_id)
            sources.append(SourceReference.from_chunk(chunk))

    return sources


def format_chunks_for_context(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks as context for the LLM.

    Args:
        chunks: List of retrieved chunks.

    Returns:
        Formatted string with source attribution.
    """
    if not chunks:
        return "No relevant sources found."

    context_parts: List[str] = []

    for i, chunk in enumerate(chunks, 1):
        source_info = f"[Source {i}: {chunk.title} ({chunk.source})]"
        context_parts.append(f"{source_info}\n{chunk.content}")

    return "\n\n---\n\n".join(context_parts)


def get_unique_article_count(chunks: List[RetrievedChunk]) -> int:
    """Count unique articles represented in the chunks."""
    return len(set(chunk.article_id for chunk in chunks))


def get_average_similarity(chunks: List[RetrievedChunk]) -> float:
    """Calculate average similarity score of chunks."""
    if not chunks:
        return 0.0
    return sum(c.similarity_score for c in chunks) / len(chunks)


def get_top_similarity(chunks: List[RetrievedChunk]) -> float:
    """Get the highest similarity score among chunks."""
    if not chunks:
        return 0.0
    return max(c.similarity_score for c in chunks)
