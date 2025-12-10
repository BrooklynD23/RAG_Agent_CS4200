from src.news_rag.core.verification import verify_summary
from src.news_rag.models.news import NewsSummary
from src.news_rag.models.news import Article
import pytest
import json
import src.news_rag.core.verification as verification


def test_verify_summary_requires_api_key(monkeypatch) -> None:
    from src.news_rag import config

    summary = NewsSummary(topic="t", summary_text="", sentences=[], sources=[])
    monkeypatch.setattr(config.settings, "google_api_key", None)
    with pytest.raises(RuntimeError):
        verify_summary(summary, [])


def test_verify_summary_uses_mocked_openai(monkeypatch) -> None:
    class DummyResponse:
        def __init__(self, content: str) -> None:
            self.text = content

    class DummyModel:
        def generate_content(self, prompt: str) -> DummyResponse:  # type: ignore[override]
            payload = {
                "overall_verdict": "supported",
                "per_sentence": [],
            }
            return DummyResponse(json.dumps(payload))

    # Avoid using a real API key or real OpenAI client.
    monkeypatch.setattr(verification, "_get_gemini_model", lambda: DummyModel())

    summary = NewsSummary(topic="t", summary_text="text", sentences=[], sources=[])
    articles = [
        Article(
            id="1",
            title="Title",
            url="https://example.com",
            source="example.com",
            content="Body",
        )
    ]

    result = verify_summary(summary, articles)
    assert result["overall_verdict"] == "supported"
