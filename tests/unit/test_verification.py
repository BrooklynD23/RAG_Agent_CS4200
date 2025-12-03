from src.news_rag.core.verification import verify_summary
from src.news_rag.models.news import NewsSummary
from src.news_rag.models.news import Article
import pytest
import json
import src.news_rag.core.verification as verification


def test_verify_summary_requires_api_key(monkeypatch) -> None:
    from src.news_rag import config

    summary = NewsSummary(topic="t", summary_text="", sentences=[], sources=[])
    monkeypatch.setattr(config.settings, "openai_api_key", None)
    with pytest.raises(RuntimeError):
        verify_summary(summary, [])


def test_verify_summary_uses_mocked_openai(monkeypatch) -> None:
    class DummyCompletions:
        def create(self, model, messages):
            payload = {
                "overall_verdict": "supported",
                "per_sentence": [],
            }

            class DummyMessage:
                def __init__(self, content: str) -> None:
                    self.content = content

            class DummyChoice:
                def __init__(self, content: str) -> None:
                    self.message = DummyMessage(content)

            class DummyResponse:
                def __init__(self, content: str) -> None:
                    self.choices = [DummyChoice(content)]

            return DummyResponse(json.dumps(payload))

    class DummyChat:
        def __init__(self) -> None:
            self.completions = DummyCompletions()

    class DummyClient:
        def __init__(self) -> None:
            self.chat = DummyChat()

    # Avoid using a real API key or real OpenAI client.
    monkeypatch.setattr(verification, "_get_openai_client", lambda: DummyClient())

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
