"""Sufficiency checker for determining if retrieved chunks can answer a query.

This module implements heuristics and optional LLM-based checks to determine
whether the retrieved chunks contain enough information to answer the user's
question, or if web search fallback is needed.
"""

import re
from typing import List, Tuple

import google.generativeai as genai

from ..config import settings
from ..logging_config import get_logger
from ..models.rag_state import RetrievedChunk

logger = get_logger("core.sufficiency_checker")

# Thresholds for heuristic checks
MIN_CHUNKS_THRESHOLD = 2
MIN_AVG_SIMILARITY_THRESHOLD = 0.35
MIN_TOP_SIMILARITY_THRESHOLD = 0.45
MIN_CONTENT_LENGTH = 200


def _extract_key_entities(query: str) -> List[str]:
    """Extract potential key entities from the query.

    Simple heuristic: extract capitalized words and quoted phrases.
    """
    entities: List[str] = []

    # Extract quoted phrases
    quoted = re.findall(r'"([^"]+)"', query)
    entities.extend(quoted)

    # Extract capitalized words (potential proper nouns)
    # Exclude common sentence starters
    words = query.split()
    for i, word in enumerate(words):
        # Skip first word of sentence
        if i > 0 and word[0].isupper() and len(word) > 2:
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word and clean_word not in ["The", "What", "How", "Why", "When", "Where"]:
                entities.append(clean_word)

    return entities


def _check_entity_coverage(query: str, chunks: List[RetrievedChunk]) -> Tuple[bool, List[str]]:
    """Check if key entities from the query appear in the chunks.

    Returns:
        Tuple of (all_covered, missing_entities)
    """
    entities = _extract_key_entities(query)

    if not entities:
        # No specific entities to check
        return True, []

    # Combine all chunk content
    combined_content = " ".join(c.content.lower() for c in chunks)

    missing = []
    for entity in entities:
        if entity.lower() not in combined_content:
            missing.append(entity)

    all_covered = len(missing) == 0
    return all_covered, missing


def _check_temporal_relevance(query: str, chunks: List[RetrievedChunk]) -> bool:
    """Check if the query asks about recent events and chunks are recent enough.

    Simple heuristic: if query contains temporal markers like "today", "this week",
    "latest", etc., check if we have recent chunks.
    """
    temporal_markers = [
        "today", "yesterday", "this week", "this month",
        "latest", "recent", "just", "breaking", "now",
        "current", "new", "update"
    ]

    query_lower = query.lower()
    has_temporal_marker = any(marker in query_lower for marker in temporal_markers)

    if not has_temporal_marker:
        return True  # No temporal requirement

    # If we have temporal markers but no chunks with dates, we can't verify
    # In this case, we'll be conservative and say it might not be sufficient
    chunks_with_dates = [c for c in chunks if c.published_at is not None]

    if not chunks_with_dates:
        # No date info, assume we might need fresh data
        return False

    # For now, just check that we have some chunks - more sophisticated
    # date checking could be added here
    return len(chunks_with_dates) > 0


def check_sufficiency_heuristic(
    query: str,
    chunks: List[RetrievedChunk],
) -> Tuple[bool, str]:
    """Check if retrieved chunks are sufficient using heuristics.

    This is a fast, non-LLM check that uses multiple signals:
    1. Number of retrieved chunks
    2. Similarity scores
    3. Total content length
    4. Entity coverage
    5. Temporal relevance

    Args:
        query: The user's question.
        chunks: Retrieved chunks from the vector store.

    Returns:
        Tuple of (is_sufficient, reason)
    """
    # Check 1: Minimum number of chunks
    if len(chunks) < MIN_CHUNKS_THRESHOLD:
        return False, f"Only {len(chunks)} chunks retrieved (need at least {MIN_CHUNKS_THRESHOLD})"

    # Check 2: Similarity scores
    avg_similarity = sum(c.similarity_score for c in chunks) / len(chunks) if chunks else 0
    top_similarity = max(c.similarity_score for c in chunks) if chunks else 0

    if top_similarity < MIN_TOP_SIMILARITY_THRESHOLD:
        return False, f"Top similarity {top_similarity:.2f} below threshold {MIN_TOP_SIMILARITY_THRESHOLD}"

    if avg_similarity < MIN_AVG_SIMILARITY_THRESHOLD:
        return False, f"Average similarity {avg_similarity:.2f} below threshold {MIN_AVG_SIMILARITY_THRESHOLD}"

    # Check 3: Total content length
    total_content = sum(len(c.content) for c in chunks)
    if total_content < MIN_CONTENT_LENGTH:
        return False, f"Total content length {total_content} below threshold {MIN_CONTENT_LENGTH}"

    # Check 4: Entity coverage
    entities_covered, missing_entities = _check_entity_coverage(query, chunks)
    if not entities_covered and len(missing_entities) > 0:
        return False, f"Missing key entities: {', '.join(missing_entities[:3])}"

    # Check 5: Temporal relevance
    if not _check_temporal_relevance(query, chunks):
        return False, "Query requires recent information but chunks may be outdated"

    return True, "Sufficient chunks with good similarity and coverage"


def check_sufficiency_llm(
    query: str,
    chunks: List[RetrievedChunk],
) -> Tuple[bool, str]:
    """Check if retrieved chunks are sufficient using an LLM.

    This is a more accurate but slower check that asks the LLM to evaluate
    whether the provided context can answer the question.

    Args:
        query: The user's question.
        chunks: Retrieved chunks from the vector store.

    Returns:
        Tuple of (is_sufficient, reason)
    """
    if not chunks:
        return False, "No chunks retrieved"

    if not settings.google_api_key:
        # Fall back to heuristic if no API key
        return check_sufficiency_heuristic(query, chunks)

    # Format chunks for the prompt
    context_parts = []
    for i, chunk in enumerate(chunks[:5], 1):  # Limit to top 5 for efficiency
        context_parts.append(f"[{i}] {chunk.content[:500]}")  # Truncate long chunks

    context = "\n\n".join(context_parts)

    system_prompt = """You are a helpful assistant that evaluates whether provided context can answer a question.

Respond with ONLY a JSON object in this exact format:
{"sufficient": true/false, "reason": "brief explanation"}

Rules:
- "sufficient": true if the context contains enough information to answer the question accurately
- "sufficient": false if the context is missing key information, is off-topic, or is too vague
- Be conservative: if unsure, say false"""

    user_prompt = f"""Question: {query}

Context:
{context}

Can this context sufficiently answer the question?"""

    try:
        genai.configure(api_key=settings.google_api_key)
        model_name = settings.news_rag_model_name or settings.google_chat_model
        model = genai.GenerativeModel(model_name)
        prompt = "\n".join(
            [
                "SYSTEM: " + system_prompt,
                "USER: " + user_prompt,
            ]
        )
        response = model.generate_content(prompt)
        content = response.text or ""

        # Parse the JSON response
        import json
        try:
            result = json.loads(content)
            is_sufficient = result.get("sufficient", False)
            reason = result.get("reason", "No reason provided")
            return is_sufficient, reason
        except json.JSONDecodeError:
            # If parsing fails, fall back to heuristic
            logger.warning("llm_sufficiency_parse_failed", content=content[:100])
            return check_sufficiency_heuristic(query, chunks)

    except Exception as exc:
        logger.error("llm_sufficiency_check_failed", error=str(exc))
        # Fall back to heuristic on error
        return check_sufficiency_heuristic(query, chunks)


def check_sufficiency(
    query: str,
    chunks: List[RetrievedChunk],
    use_llm: bool = False,
) -> Tuple[bool, str]:
    """Main entry point for sufficiency checking.

    Args:
        query: The user's question.
        chunks: Retrieved chunks from the vector store.
        use_llm: Whether to use LLM-based checking (slower but more accurate).

    Returns:
        Tuple of (is_sufficient, reason)
    """
    if use_llm:
        return check_sufficiency_llm(query, chunks)
    else:
        return check_sufficiency_heuristic(query, chunks)
