from typing import Any, Dict, List
import json

import google.generativeai as genai

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


def _get_gemini_model() -> genai.GenerativeModel:
    """Return a Gemini model client using configuration from settings."""

    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured in the environment.")
    genai.configure(api_key=settings.google_api_key)
    model_name = settings.news_rag_model_name or settings.google_chat_model
    return genai.GenerativeModel(model_name)


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

    prompt = "\n\n".join(
        [
            "SYSTEM: " + SUMMARIZER_SYSTEM_PROMPT,
            "USER: " + json.dumps(payload),
        ]
    )

    model = _get_gemini_model()
    response = model.generate_content(prompt)

    try:
        content = response.text or ""
    except Exception as exc:
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
