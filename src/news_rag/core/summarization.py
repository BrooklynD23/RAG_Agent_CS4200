from typing import Any, Dict, List
import json

from openai import OpenAI

from ..config import settings
from ..logging_config import get_logger
from ..models.news import Article, NewsSummary, SummarySentence
from .prompts import SUMMARIZER_SYSTEM_PROMPT


logger = get_logger("core.summarization")


def build_summarizer_input(topic: str, articles: List[Article]) -> Dict[str, Any]:
    """Build the structured input payload for the summarizer LLM.

    This does not call any external APIs; it only prepares the
    information the LLM needs, following the architecture docs.
    """

    return {
        "topic": topic,
        "articles": [
            {
                "id": article.id,
                "title": article.title,
                "source": article.source,
                "url": article.url,
                "content": article.content,
            }
            for article in articles
        ],
    }


def _get_openai_client() -> OpenAI:
    """Return an OpenAI client using configuration from settings."""

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured in the environment.")
    return OpenAI(api_key=settings.openai_api_key)


def summarize_articles(topic: str, articles: List[Article]) -> NewsSummary:
    """Summarize a list of articles into a NewsSummary using OpenAI.

    See `docs/architecture/04_generation-and-prompting.md` for details.
    """

    if not articles:
        logger.info(
            "summarize_articles_no_articles",
            topic=topic,
        )
        return NewsSummary(
            topic=topic,
            summary_text="No relevant articles were retrieved for this topic.",
            sentences=[],
            sources=[],
            meta={"warning": "no_articles"},
        )

    client = _get_openai_client()
    payload = build_summarizer_input(topic, articles)
    messages = [
        {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload)},
    ]

    logger.info(
        "summarize_articles_call",
        topic=topic,
        articles=len(articles),
        model=settings.news_rag_model_name,
    )

    response = client.chat.completions.create(
        model=settings.news_rag_model_name,
        messages=messages,
    )

    try:
        content = response.choices[0].message.content or ""
    except (AttributeError, IndexError) as exc:
        logger.warning("summarize_articles_invalid_response", error=str(exc))
        raise RuntimeError("Summarizer returned an invalid response structure.") from exc

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.warning("summarize_articles_non_json", error=str(exc))
        raise RuntimeError("Summarizer returned non-JSON content.") from exc

    if "summary_text" not in data or "sentences" not in data:
        logger.warning("summarize_articles_missing_keys")
        raise RuntimeError("Summarizer JSON missing required keys.")

    sentences = [
        SummarySentence(**s)
        for s in data.get("sentences", [])
    ]

    summary = NewsSummary(
        topic=topic,
        summary_text=data["summary_text"],
        sentences=sentences,
        sources=articles,
        meta={"model": settings.news_rag_model_name},
    )
    logger.info(
        "summarize_articles_success",
        topic=topic,
        sentences=len(summary.sentences),
    )
    return summary
