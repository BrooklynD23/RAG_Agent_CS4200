"""Integration tests for the RAG API endpoints.

These tests require the FastAPI server to be running or use TestClient.
They test the full flow from API request to response.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.news_rag.api.server import app
from src.news_rag.models.news import Article
from src.news_rag.models.rag_state import RetrievedChunk


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_articles():
    """Sample articles for mocking."""
    return [
        Article(
            id="test_1",
            title="Test Article 1",
            url="https://example.com/1",
            source="TestSource",
            published_at=datetime(2024, 1, 15),
            content="This is test content about AI regulations in Europe.",
            score=0.9,
        ),
        Article(
            id="test_2",
            title="Test Article 2",
            url="https://example.com/2",
            source="TestSource2",
            published_at=datetime(2024, 1, 16),
            content="More content about technology and regulations.",
            score=0.85,
        ),
    ]


@pytest.fixture
def sample_chunks():
    """Sample chunks for mocking retrieval."""
    return [
        RetrievedChunk(
            chunk_id="chunk_1",
            article_id="test_1",
            conversation_id="test_conv",
            content="This is test content about AI regulations in Europe.",
            chunk_index=0,
            url="https://example.com/1",
            title="Test Article 1",
            source="TestSource",
            published_at=datetime(2024, 1, 15),
            similarity_score=0.85,
        ),
    ]


class TestHealthEndpoint:
    """Tests for the health endpoint."""

    def test_health_check(self, client):
        """Test that health endpoint returns ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestLegacySummarizeEndpoint:
    """Tests for the legacy /summarize endpoint."""

    @patch("src.news_rag.api.server.retrieve_articles")
    @patch("src.news_rag.api.server.summarize_articles")
    @patch("src.news_rag.api.server.classify_query")
    def test_summarize_success(
        self,
        mock_classify,
        mock_summarize,
        mock_retrieve,
        client,
        sample_articles,
    ):
        """Test successful summarization."""
        from src.news_rag.models.news import NewsSummary

        mock_classify.return_value = "news"
        mock_retrieve.return_value = sample_articles
        mock_summarize.return_value = NewsSummary(
            topic="AI regulations",
            summary_text="Summary of AI regulations...",
            sentences=[],
            sources=sample_articles,
            meta={},
        )

        response = client.post(
            "/summarize",
            json={
                "query": "AI regulations",
                "time_range": "7d",
                "verification": False,
                "max_articles": 5,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "summary_text" in data
        assert "sources" in data


class TestRAGQueryEndpoint:
    """Tests for the RAG /rag/query endpoint."""

    @patch("src.news_rag.api.server.run_news_query")
    def test_rag_query_initial(self, mock_run_query, client):
        """Test initial RAG query."""
        from src.news_rag.models.rag_state import AgentResponse, SourceReference

        mock_run_query.return_value = AgentResponse(
            answer_text="Summary of the news...",
            answer_type="summary",
            sources=[
                SourceReference(
                    article_id="test_1",
                    url="https://example.com/1",
                    title="Test Article",
                    source="TestSource",
                    published_at=datetime(2024, 1, 15),
                )
            ],
            conversation_id="new_conv_123",
            debug=None,
        )

        response = client.post(
            "/rag/query",
            json={
                "message": "What are the latest AI news?",
                "time_range": "7d",
                "max_articles": 10,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer_type"] == "summary"
        assert data["conversation_id"] == "new_conv_123"
        assert len(data["sources"]) == 1

    @patch("src.news_rag.api.server.run_news_query")
    def test_rag_query_followup(self, mock_run_query, client):
        """Test follow-up RAG query."""
        from src.news_rag.models.rag_state import AgentResponse, SourceReference

        mock_run_query.return_value = AgentResponse(
            answer_text="Based on the sources, the answer is...",
            answer_type="followup_answer",
            sources=[
                SourceReference(
                    article_id="test_1",
                    url="https://example.com/1",
                    title="Test Article",
                    source="TestSource",
                )
            ],
            conversation_id="existing_conv_123",
            debug={"chunks_retrieved": 3},
        )

        response = client.post(
            "/rag/query",
            json={
                "message": "Can you explain more about that?",
                "conversation_id": "existing_conv_123",
                "include_debug": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer_type"] == "followup_answer"
        assert data["conversation_id"] == "existing_conv_123"
        assert data["debug"] is not None

    @patch("src.news_rag.api.server.run_news_query")
    def test_rag_query_web_augmented(self, mock_run_query, client):
        """Test RAG query with web search fallback."""
        from src.news_rag.models.rag_state import AgentResponse, SourceReference

        mock_run_query.return_value = AgentResponse(
            answer_text="After searching for more information...",
            answer_type="web_augmented_answer",
            sources=[
                SourceReference(
                    article_id="new_1",
                    url="https://example.com/new",
                    title="New Article",
                    source="NewSource",
                )
            ],
            conversation_id="conv_123",
        )

        response = client.post(
            "/rag/query",
            json={
                "message": "What about something not in the original articles?",
                "conversation_id": "conv_123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer_type"] == "web_augmented_answer"


class TestRAGConversationEndpoints:
    """Tests for conversation management endpoints."""

    @patch("src.news_rag.api.server.get_conversation_sources")
    def test_get_conversation_sources(self, mock_get_sources, client):
        """Test getting sources for a conversation."""
        from src.news_rag.models.rag_state import SourceReference

        mock_get_sources.return_value = [
            SourceReference(
                article_id="test_1",
                url="https://example.com/1",
                title="Test Article",
                source="TestSource",
            ),
        ]

        response = client.get("/rag/conversation/conv_123/sources")

        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == "conv_123"
        assert data["count"] == 1
        assert len(data["sources"]) == 1

    @patch("src.news_rag.api.server.clear_conversation")
    def test_delete_conversation(self, mock_clear, client):
        """Test deleting a conversation."""
        mock_clear.return_value = 10  # 10 chunks deleted

        response = client.delete("/rag/conversation/conv_123")

        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == "conv_123"
        assert data["chunks_deleted"] == 10
        assert data["status"] == "deleted"


class TestRAGStatsEndpoint:
    """Tests for the stats endpoint."""

    @patch("src.news_rag.api.server.get_collection_stats")
    def test_get_stats(self, mock_stats, client):
        """Test getting vector store stats."""
        mock_stats.return_value = {
            "name": "news_articles",
            "count": 100,
            "persist_dir": ".chroma_db",
        }

        response = client.get("/rag/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "vector_store" in data
        assert data["vector_store"]["count"] == 100


class TestErrorHandling:
    """Tests for error handling in API endpoints."""

    @patch("src.news_rag.api.server.run_news_query")
    def test_rag_query_error(self, mock_run_query, client):
        """Test error handling in RAG query."""
        mock_run_query.side_effect = Exception("Test error")

        response = client.post(
            "/rag/query",
            json={"message": "Test query"},
        )

        assert response.status_code == 500
        assert "Test error" in response.json()["detail"]

    @patch("src.news_rag.api.server.get_conversation_sources")
    def test_get_sources_error(self, mock_get_sources, client):
        """Test error handling in get sources."""
        mock_get_sources.side_effect = Exception("Database error")

        response = client.get("/rag/conversation/conv_123/sources")

        assert response.status_code == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
