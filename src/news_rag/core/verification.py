from typing import Any, Dict, List
import json

import google.generativeai as genai

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


def _get_gemini_model() -> genai.GenerativeModel:
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured in the environment.")
    genai.configure(api_key=settings.google_api_key)
    model_name = settings.news_rag_model_name or settings.google_chat_model
    return genai.GenerativeModel(model_name)


def verify_summary(summary: NewsSummary, articles: List[Article]) -> Dict[str, Any]:
    """Run an optional critic/verification pass over a summary.

    Returns the parsed JSON verdict from the critic model.
    """

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

    prompt = "\n\n".join(
        [
            "SYSTEM: " + CRITIC_SYSTEM_PROMPT,
            "USER: " + json.dumps(payload),
        ]
    )

    model = _get_gemini_model()
    response = model.generate_content(prompt)

    try:
        content = response.text or ""
    except Exception as exc:
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
