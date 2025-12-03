from typing import Any, Dict, List
import json

from openai import OpenAI

from ..config import settings
from ..logging_config import get_logger
from ..models.news import Article, NewsSummary
from .prompts import CRITIC_SYSTEM_PROMPT


logger = get_logger("core.verification")


def build_verifier_input(summary: NewsSummary, articles: List[Article]) -> Dict[str, Any]:
    """Build the structured input payload for the critic/verification LLM."""

    return {
        "summary_text": summary.summary_text,
        "sentences": [
            {"text": s.text, "source_ids": s.source_ids}
            for s in summary.sentences
        ],
        "articles": [
            {
                "id": a.id,
                "title": a.title,
                "source": a.source,
                "url": a.url,
                "content": a.content,
            }
            for a in articles
        ],
    }


def _get_openai_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured in the environment.")
    return OpenAI(api_key=settings.openai_api_key)


def verify_summary(summary: NewsSummary, articles: List[Article]) -> Dict[str, Any]:
    """Run an optional critic/verification pass over a summary.

    Returns the parsed JSON verdict from the critic model.
    """

    client = _get_openai_client()
    payload = build_verifier_input(summary, articles)
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload)},
    ]

    logger.info(
        "verify_summary_call",
        summary_topic=summary.topic,
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
        logger.warning("verify_summary_invalid_response", error=str(exc))
        raise RuntimeError("Critic returned an invalid response structure.") from exc

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.warning("verify_summary_non_json", error=str(exc))
        raise RuntimeError("Critic returned non-JSON content.") from exc

    if "overall_verdict" not in data:
        logger.warning("verify_summary_missing_keys")
        raise RuntimeError("Critic JSON missing required keys.")

    logger.info("verify_summary_success", summary_topic=summary.topic)
    return data
