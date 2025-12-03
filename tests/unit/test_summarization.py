from src.news_rag.core.summarization import build_summarizer_input, summarize_articles
from src.news_rag.models.news import Article
import pytest
import json
import src.news_rag.core.summarization as summarization


def test_build_summarizer_input_structure() -> None:
    articles = [
        Article(
            id="1",
            title="Title",
            url="https://example.com",
            source="example.com",
            content="Body",
        )
    ]
    payload = build_summarizer_input("topic", articles)
    assert payload["topic"] == "topic"
    assert payload["articles"][0]["id"] == "1"


def test_summarize_articles_requires_api_key(monkeypatch) -> None:
    from src.news_rag import config

    articles = [
        Article(
            id="1",
            title="Title",
            url="https://example.com",
            source="example.com",
            content="Body",
        )
    ]

    monkeypatch.setattr(config.settings, "openai_api_key", None)
    with pytest.raises(RuntimeError):
        summarize_articles("topic", articles)


def test_summarize_articles_uses_mocked_openai(monkeypatch) -> None:
    class DummyCompletions:
        def create(self, model, messages):
            # The implementation should send our payload as JSON; we return
            # a minimal JSON response that the summarization code expects.
            payload = {
                "summary_text": "Stub summary",
                "sentences": [
                    {"text": "Stub sentence", "source_ids": ["1"]},
                ],
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
    monkeypatch.setattr(summarization, "_get_openai_client", lambda: DummyClient())

    articles = [
        Article(
            id="1",
            title="Title",
            url="https://example.com",
            source="example.com",
            content="Body",
        )
    ]

    summary = summarize_articles("topic", articles)
    assert summary.summary_text == "Stub summary"
    assert len(summary.sentences) == 1
    assert summary.sentences[0].text == "Stub sentence"
    # Sources should be carried through unchanged.
    assert len(summary.sources) == 1
    assert summary.sources[0].id == "1"
