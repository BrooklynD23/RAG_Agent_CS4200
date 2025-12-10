"""Vector store management using ChromaDB.

This module provides a singleton ChromaDB collection for storing and
retrieving article chunks with their embeddings. It uses OpenAI embeddings
via LangChain for consistency with the rest of the pipeline.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..config import settings
from ..logging_config import get_logger
from ..models.rag_state import ArticleChunk, RetrievedChunk

logger = get_logger("core.vector_store")

# Singleton instances
_chroma_client: Optional[chromadb.Client] = None
_collection: Optional[chromadb.Collection] = None
_embeddings: Optional[GoogleGenerativeAIEmbeddings] = None

# Collection name for news articles
COLLECTION_NAME = "news_articles"

# Persist directory for ChromaDB (relative to project root)
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma_db")


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Get or create the embeddings instance."""
    global _embeddings
    if _embeddings is None:
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is not configured.")
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.google_embedding_model,
            google_api_key=settings.google_api_key,
        )
    return _embeddings


def _get_chroma_client() -> chromadb.Client:
    """Get or create the ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        # Use persistent storage so data survives restarts
        # Note: Using PersistentClient for newer ChromaDB versions
        try:
            _chroma_client = chromadb.PersistentClient(
                path=PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        except TypeError:
            # Fallback for older ChromaDB versions
            _chroma_client = chromadb.Client(
                ChromaSettings(
                    persist_directory=PERSIST_DIR,
                    anonymized_telemetry=False,
                )
            )
        logger.info("chroma_client_initialized", persist_dir=PERSIST_DIR)
    return _chroma_client


def get_collection() -> chromadb.Collection:
    """Get or create the news articles collection."""
    global _collection
    if _collection is None:
        client = _get_chroma_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "News article chunks for RAG"},
        )
        logger.info(
            "collection_initialized",
            name=COLLECTION_NAME,
            count=_collection.count(),
        )
    return _collection


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using OpenAI embeddings."""
    embeddings = _get_embeddings()
    return embeddings.embed_documents(texts)


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    embeddings = _get_embeddings()
    return embeddings.embed_query(query)


def add_chunks(chunks: List[ArticleChunk]) -> int:
    """Add article chunks to the vector store.

    Args:
        chunks: List of ArticleChunk objects to store.

    Returns:
        Number of chunks successfully added.
    """
    if not chunks:
        return 0

    collection = get_collection()

    # Prepare data for ChromaDB
    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.content for chunk in chunks]
    metadatas = [chunk.to_metadata() for chunk in chunks]

    # Generate embeddings
    try:
        embeddings = embed_texts(documents)
    except Exception as exc:
        logger.error("embedding_failed", error=str(exc), chunk_count=len(chunks))
        raise RuntimeError(f"Failed to generate embeddings: {exc}") from exc

    # Add to collection
    try:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(
            "chunks_added",
            count=len(chunks),
            conversation_id=chunks[0].conversation_id if chunks else None,
        )
        return len(chunks)
    except Exception as exc:
        logger.error("add_chunks_failed", error=str(exc))
        raise RuntimeError(f"Failed to add chunks to vector store: {exc}") from exc


def query_chunks(
    query: str,
    conversation_id: Optional[str] = None,
    n_results: int = 10,
    similarity_threshold: float = 0.0,
) -> List[RetrievedChunk]:
    """Query the vector store for relevant chunks.

    Args:
        query: The search query.
        conversation_id: Optional filter to only search within a conversation.
        n_results: Maximum number of results to return.
        similarity_threshold: Minimum similarity score (0-1) to include.

    Returns:
        List of RetrievedChunk objects sorted by similarity.
    """
    collection = get_collection()

    # Embed the query
    try:
        query_embedding = embed_query(query)
    except Exception as exc:
        logger.error("query_embedding_failed", error=str(exc))
        raise RuntimeError(f"Failed to embed query: {exc}") from exc

    # Build where filter for conversation_id
    where_filter = None
    if conversation_id:
        where_filter = {"conversation_id": conversation_id}

    # Query the collection
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error("query_failed", error=str(exc))
        raise RuntimeError(f"Failed to query vector store: {exc}") from exc

    # Convert results to RetrievedChunk objects
    retrieved_chunks: List[RetrievedChunk] = []

    if not results or not results.get("ids") or not results["ids"][0]:
        return retrieved_chunks

    ids = results["ids"][0]
    documents = results["documents"][0] if results.get("documents") else []
    metadatas = results["metadatas"][0] if results.get("metadatas") else []
    distances = results["distances"][0] if results.get("distances") else []

    for i, chunk_id in enumerate(ids):
        # ChromaDB returns L2 distance; convert to similarity score
        # Lower distance = higher similarity
        distance = distances[i] if i < len(distances) else 1.0
        # Approximate conversion: similarity = 1 / (1 + distance)
        similarity = 1.0 / (1.0 + distance)

        if similarity < similarity_threshold:
            continue

        metadata = metadatas[i] if i < len(metadatas) else {}
        content = documents[i] if i < len(documents) else ""

        # Parse published_at from metadata
        published_at = None
        if metadata.get("published_at"):
            try:
                published_at = datetime.fromisoformat(metadata["published_at"])
            except (ValueError, TypeError):
                pass

        retrieved_chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                article_id=metadata.get("article_id", ""),
                conversation_id=metadata.get("conversation_id", ""),
                content=content,
                chunk_index=metadata.get("chunk_index", 0),
                url=metadata.get("url", ""),
                title=metadata.get("title", ""),
                source=metadata.get("source", ""),
                published_at=published_at,
                similarity_score=similarity,
            )
        )

    # Sort by similarity (highest first)
    retrieved_chunks.sort(key=lambda x: x.similarity_score, reverse=True)

    logger.info(
        "chunks_retrieved",
        query_preview=query[:50],
        conversation_id=conversation_id,
        results=len(retrieved_chunks),
    )

    return retrieved_chunks


def get_chunks_by_conversation(conversation_id: str) -> List[RetrievedChunk]:
    """Get all chunks for a specific conversation.

    Args:
        conversation_id: The conversation ID to filter by.

    Returns:
        List of all chunks in the conversation.
    """
    collection = get_collection()

    try:
        results = collection.get(
            where={"conversation_id": conversation_id},
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        logger.error("get_by_conversation_failed", error=str(exc))
        return []

    chunks: List[RetrievedChunk] = []

    if not results or not results.get("ids"):
        return chunks

    ids = results["ids"]
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    for i, chunk_id in enumerate(ids):
        metadata = metadatas[i] if i < len(metadatas) else {}
        content = documents[i] if i < len(documents) else ""

        published_at = None
        if metadata.get("published_at"):
            try:
                published_at = datetime.fromisoformat(metadata["published_at"])
            except (ValueError, TypeError):
                pass

        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                article_id=metadata.get("article_id", ""),
                conversation_id=metadata.get("conversation_id", ""),
                content=content,
                chunk_index=metadata.get("chunk_index", 0),
                url=metadata.get("url", ""),
                title=metadata.get("title", ""),
                source=metadata.get("source", ""),
                published_at=published_at,
                similarity_score=1.0,  # Not from query, so no score
            )
        )

    return chunks


def delete_conversation_chunks(conversation_id: str) -> int:
    """Delete all chunks for a specific conversation.

    Args:
        conversation_id: The conversation ID to delete.

    Returns:
        Number of chunks deleted.
    """
    collection = get_collection()

    # First get the IDs to delete
    try:
        results = collection.get(
            where={"conversation_id": conversation_id},
            include=[],
        )
    except Exception as exc:
        logger.error("get_for_delete_failed", error=str(exc))
        return 0

    if not results or not results.get("ids"):
        return 0

    ids_to_delete = results["ids"]

    try:
        collection.delete(ids=ids_to_delete)
        logger.info(
            "chunks_deleted",
            conversation_id=conversation_id,
            count=len(ids_to_delete),
        )
        return len(ids_to_delete)
    except Exception as exc:
        logger.error("delete_failed", error=str(exc))
        return 0


def get_collection_stats() -> Dict[str, Any]:
    """Get statistics about the vector store collection."""
    collection = get_collection()
    return {
        "name": COLLECTION_NAME,
        "count": collection.count(),
        "persist_dir": PERSIST_DIR,
    }


def clear_collection() -> None:
    """Clear all data from the collection. Use with caution."""
    global _collection
    client = _get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        _collection = None
        logger.warning("collection_cleared", name=COLLECTION_NAME)
    except Exception as exc:
        logger.error("clear_collection_failed", error=str(exc))
