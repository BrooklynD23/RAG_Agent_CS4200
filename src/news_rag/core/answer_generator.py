"""Answer generation for RAG queries.

This module generates grounded answers from retrieved chunks,
with explicit source citations and the ability to indicate
when information is not available.
"""

import json
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from ..config import settings
from ..logging_config import get_logger
from ..models.rag_state import RetrievedChunk, SourceReference

logger = get_logger("core.answer_generator")


ANSWER_SYSTEM_PROMPT = """You are a helpful news assistant that answers questions based on provided source material.

CRITICAL RULES:
1. ONLY use information from the provided sources. Do not use prior knowledge.
2. Cite sources using [Source N] format after each claim.
3. If the sources don't contain enough information to fully answer the question, explicitly say so.
4. If sources disagree, present both viewpoints with their respective citations.
5. Be concise but thorough.
6. Never make up information not in the sources.

You MUST respond with valid JSON in this exact format:
{
    "answer": "Your answer text with [Source N] citations",
    "sources_used": [1, 2, 3],
    "confidence": "high" | "medium" | "low",
    "missing_info": "Description of what info is missing, or null if complete"
}"""


FOLLOWUP_SYSTEM_PROMPT = """You are a helpful news assistant answering follow-up questions about previously retrieved news articles.

CRITICAL RULES:
1. ONLY use information from the provided sources. Do not use prior knowledge.
2. Cite sources using [Source N] format after each claim.
3. If the sources don't contain the answer, say "Based on the available sources, I don't have information about..."
4. You can explain, expand on, or clarify information from the sources.
5. You can provide direct quotes from sources when relevant.
6. Be conversational but accurate.

You MUST respond with valid JSON in this exact format:
{
    "answer": "Your answer text with [Source N] citations",
    "sources_used": [1, 2, 3],
    "confidence": "high" | "medium" | "low",
    "missing_info": "Description of what info is missing, or null if complete"
}"""


def _format_sources_for_prompt(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks as numbered sources for the prompt."""
    if not chunks:
        return "No sources available."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[Source {i}]"
        metadata = f"Title: {chunk.title}\nPublisher: {chunk.source}\nURL: {chunk.url}"
        content = f"Content:\n{chunk.content}"
        parts.append(f"{header}\n{metadata}\n{content}")

    return "\n\n---\n\n".join(parts)


def _get_gemini_model() -> genai.GenerativeModel:
    """Get Gemini model client."""
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured.")
    genai.configure(api_key=settings.google_api_key)
    model_name = settings.news_rag_model_name or settings.google_chat_model
    return genai.GenerativeModel(model_name)


def _parse_answer_response(content: str) -> Dict[str, Any]:
    """Parse the JSON response from the answer generator."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Return a fallback structure
        return {
            "answer": content,
            "sources_used": [],
            "confidence": "low",
            "missing_info": "Response parsing failed",
        }


def generate_answer(
    query: str,
    chunks: List[RetrievedChunk],
    is_followup: bool = False,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Generate a grounded answer from retrieved chunks.

    Args:
        query: The user's question.
        chunks: Retrieved chunks to use as sources.
        is_followup: Whether this is a follow-up question.
        conversation_history: Optional previous messages for context.

    Returns:
        Dict with keys: answer, sources_used, confidence, missing_info
    """
    if not chunks:
        return {
            "answer": "I don't have any relevant sources to answer this question. Would you like me to search for more information?",
            "sources_used": [],
            "confidence": "low",
            "missing_info": "No sources available",
        }

    # Format sources
    sources_text = _format_sources_for_prompt(chunks)

    # Choose system prompt based on query type
    system_prompt = FOLLOWUP_SYSTEM_PROMPT if is_followup else ANSWER_SYSTEM_PROMPT

    # Build messages for prompt formatting
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Add conversation history if available (for follow-ups)
    if conversation_history and is_followup:
        # Add last few exchanges for context
        for msg in conversation_history[-4:]:
            messages.append(msg)

    # Add the current query with sources
    user_content = f"""Question: {query}

Sources:
{sources_text}

Please answer the question using only the provided sources."""

    messages.append({"role": "user", "content": user_content})

    logger.info(
        "generating_answer",
        query_preview=query[:50],
        chunks=len(chunks),
        is_followup=is_followup,
    )

    try:
        # Flatten chat messages into a single prompt for Gemini
        prompt_lines = [
            f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}"
            for msg in messages
        ]
        prompt = "\n\n".join(prompt_lines)

        model = _get_gemini_model()
        response = model.generate_content(prompt)
        content = response.text or ""
        result = _parse_answer_response(content)

        logger.info(
            "answer_generated",
            confidence=result.get("confidence"),
            sources_used=len(result.get("sources_used", [])),
        )

        return result

    except Exception as exc:
        logger.error("answer_generation_failed", error=str(exc))
        return {
            "answer": f"I encountered an error while generating the answer: {str(exc)}",
            "sources_used": [],
            "confidence": "low",
            "missing_info": str(exc),
        }


def generate_summary_answer(
    query: str,
    chunks: List[RetrievedChunk],
) -> Dict[str, Any]:
    """Generate a summary-style answer for initial queries.

    This produces a more structured summary with bullet points
    rather than a conversational answer.

    Args:
        query: The user's news query.
        chunks: Retrieved chunks to summarize.

    Returns:
        Dict with keys: answer, sources_used, confidence, missing_info
    """
    if not chunks:
        return {
            "answer": "No relevant news articles were found for this topic.",
            "sources_used": [],
            "confidence": "low",
            "missing_info": "No sources available",
        }

    sources_text = _format_sources_for_prompt(chunks)

    system_prompt = """You are a news summarization assistant that creates concise, well-cited summaries.

RULES:
1. Create a summary with 3-7 bullet points covering the key facts.
2. Each bullet point MUST have at least one [Source N] citation.
3. Only include information from the provided sources.
4. If sources disagree, note the disagreement.
5. Be factual and objective.

You MUST respond with valid JSON:
{
    "answer": "• First point [Source 1]\\n• Second point [Source 2, 3]\\n...",
    "sources_used": [1, 2, 3],
    "confidence": "high" | "medium" | "low",
    "missing_info": null or "description of gaps"
}"""

    user_content = f"""Topic: {query}

Sources:
{sources_text}

Create a bullet-point summary of the key facts about this topic."""

    try:
        prompt = "\n".join(
            [
                "SYSTEM: " + system_prompt,
                "USER: " + user_content,
            ]
        )

        model = _get_gemini_model()
        response = model.generate_content(prompt)
        content = response.text or ""
        return _parse_answer_response(content)

    except Exception as exc:
        logger.error("summary_generation_failed", error=str(exc))
        return {
            "answer": f"Error generating summary: {str(exc)}",
            "sources_used": [],
            "confidence": "low",
            "missing_info": str(exc),
        }


def map_sources_used_to_references(
    sources_used: List[int],
    chunks: List[RetrievedChunk],
) -> List[SourceReference]:
    """Map source indices from the answer to SourceReference objects.

    Args:
        sources_used: List of 1-indexed source numbers from the answer.
        chunks: The chunks that were provided as sources.

    Returns:
        List of unique SourceReference objects.
    """
    seen_article_ids = set()
    references = []

    for idx in sources_used:
        # Convert to 0-indexed
        chunk_idx = idx - 1
        if 0 <= chunk_idx < len(chunks):
            chunk = chunks[chunk_idx]
            if chunk.article_id not in seen_article_ids:
                seen_article_ids.add(chunk.article_id)
                references.append(SourceReference.from_chunk(chunk))

    return references
