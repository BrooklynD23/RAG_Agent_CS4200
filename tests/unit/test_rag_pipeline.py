"""Unit tests for the RAG pipeline components.

These tests use fixtures and mocks to test the RAG components
without requiring external API calls.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.news_rag.models.news import Article
from src.news_rag.models.rag_state import (
    ArticleChunk,
    RetrievedChunk,
    SourceReference,
    RAGState,
    AgentResponse,
    generate_id,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_articles():
    """Sample articles for testing."""
    return [
        Article(
            id="article_1",
            title="AI Regulation in Europe",
            url="https://example.com/ai-regulation",
            source="TechNews",
            published_at=datetime(2024, 1, 15, 10, 0, 0),
            content="The European Union has proposed new regulations for artificial intelligence. "
                    "The AI Act aims to establish a legal framework for AI systems. "
                    "High-risk AI applications will face stricter requirements.",
            score=0.95,
        ),
        Article(
            id="article_2",
            title="Global Tech Companies Respond to AI Rules",
            url="https://example.com/tech-response",
            source="BusinessDaily",
            published_at=datetime(2024, 1, 16, 14, 30, 0),
            content="Major technology companies have expressed mixed reactions to the proposed AI regulations. "
                    "Some support the framework while others argue it may stifle innovation. "
                    "Industry groups are lobbying for amendments.",
            score=0.88,
        ),
        Article(
            id="article_3",
            title="AI Safety Research Advances",
            url="https://example.com/ai-safety",
            source="ScienceWeekly",
            published_at=datetime(2024, 1, 17, 9, 15, 0),
            content="Researchers have made significant progress in AI safety techniques. "
                    "New methods for aligning AI systems with human values show promise. "
                    "The field continues to grow as AI capabilities advance.",
            score=0.82,
        ),
    ]


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing retrieval."""
    conversation_id = "test_conv_123"
    return [
        RetrievedChunk(
            chunk_id="chunk_1_0",
            article_id="article_1",
            conversation_id=conversation_id,
            content="The European Union has proposed new regulations for artificial intelligence.",
            chunk_index=0,
            url="https://example.com/ai-regulation",
            title="AI Regulation in Europe",
            source="TechNews",
            published_at=datetime(2024, 1, 15, 10, 0, 0),
            similarity_score=0.85,
        ),
        RetrievedChunk(
            chunk_id="chunk_1_1",
            article_id="article_1",
            conversation_id=conversation_id,
            content="The AI Act aims to establish a legal framework for AI systems.",
            chunk_index=1,
            url="https://example.com/ai-regulation",
            title="AI Regulation in Europe",
            source="TechNews",
            published_at=datetime(2024, 1, 15, 10, 0, 0),
            similarity_score=0.78,
        ),
        RetrievedChunk(
            chunk_id="chunk_2_0",
            article_id="article_2",
            conversation_id=conversation_id,
            content="Major technology companies have expressed mixed reactions to the proposed AI regulations.",
            chunk_index=0,
            url="https://example.com/tech-response",
            title="Global Tech Companies Respond to AI Rules",
            source="BusinessDaily",
            published_at=datetime(2024, 1, 16, 14, 30, 0),
            similarity_score=0.72,
        ),
    ]


# ============================================================================
# Model Tests
# ============================================================================


class TestRAGStateModels:
    """Tests for RAG state models."""

    def test_generate_id(self):
        """Test ID generation produces unique IDs."""
        id1 = generate_id()
        id2 = generate_id()
        assert id1 != id2
        assert len(id1) == 12
        assert len(id2) == 12

    def test_article_chunk_to_metadata(self, sample_articles):
        """Test ArticleChunk metadata conversion."""
        article = sample_articles[0]
        chunk = ArticleChunk(
            chunk_id="test_chunk",
            article_id=article.id,
            conversation_id="conv_123",
            content="Test content",
            chunk_index=0,
            url=article.url,
            title=article.title,
            source=article.source,
            published_at=article.published_at,
        )

        metadata = chunk.to_metadata()

        assert metadata["chunk_id"] == "test_chunk"
        assert metadata["article_id"] == "article_1"
        assert metadata["conversation_id"] == "conv_123"
        assert metadata["url"] == article.url
        assert metadata["title"] == article.title

    def test_source_reference_from_chunk(self, sample_chunks):
        """Test SourceReference creation from chunk."""
        chunk = sample_chunks[0]
        source_ref = SourceReference.from_chunk(chunk)

        assert source_ref.article_id == chunk.article_id
        assert source_ref.url == chunk.url
        assert source_ref.title == chunk.title
        assert source_ref.source == chunk.source

    def test_rag_state_defaults(self):
        """Test RAGState default values."""
        state = RAGState(query="test query")

        assert state.query == "test query"
        assert state.message_type == "initial"
        assert state.conversation_id is not None
        assert state.status == "init"
        assert state.articles == []
        assert state.retrieved_chunks == []

    def test_agent_response_from_state(self, sample_chunks):
        """Test AgentResponse creation from RAGState."""
        state = RAGState(
            query="test query",
            conversation_id="conv_123",
            answer_text="Test answer",
            answer_type="summary",
            sources_used=[SourceReference.from_chunk(sample_chunks[0])],
            debug_info={"test": "info"},
        )

        response = AgentResponse.from_state(state, include_debug=True)

        assert response.answer_text == "Test answer"
        assert response.answer_type == "summary"
        assert response.conversation_id == "conv_123"
        assert len(response.sources) == 1
        assert response.debug == {"test": "info"}


# ============================================================================
# Sufficiency Checker Tests
# ============================================================================


class TestSufficiencyChecker:
    """Tests for the sufficiency checker."""

    def test_insufficient_when_no_chunks(self):
        """Test that empty chunks are marked insufficient."""
        from src.news_rag.core.sufficiency_checker import check_sufficiency_heuristic

        is_sufficient, reason = check_sufficiency_heuristic(
            query="What are the latest AI regulations?",
            chunks=[],
        )

        assert not is_sufficient
        assert "chunks" in reason.lower()

    def test_insufficient_when_low_similarity(self, sample_chunks):
        """Test that low similarity scores are marked insufficient."""
        from src.news_rag.core.sufficiency_checker import check_sufficiency_heuristic

        # Modify chunks to have low similarity
        low_sim_chunks = []
        for chunk in sample_chunks:
            modified = chunk.model_copy(update={"similarity_score": 0.2})
            low_sim_chunks.append(modified)

        is_sufficient, reason = check_sufficiency_heuristic(
            query="What are the latest AI regulations?",
            chunks=low_sim_chunks,
        )

        assert not is_sufficient
        assert "similarity" in reason.lower()

    def test_sufficient_with_good_chunks(self, sample_chunks):
        """Test that good chunks are marked sufficient."""
        from src.news_rag.core.sufficiency_checker import check_sufficiency_heuristic

        is_sufficient, reason = check_sufficiency_heuristic(
            query="What are the latest AI regulations?",
            chunks=sample_chunks,
        )

        assert is_sufficient

    def test_entity_coverage_check(self, sample_chunks):
        """Test entity coverage detection."""
        from src.news_rag.core.sufficiency_checker import _check_entity_coverage

        # Query with entity present in chunks
        covered, missing = _check_entity_coverage(
            query="What is the European Union doing about AI?",
            chunks=sample_chunks,
        )
        assert covered or len(missing) == 0 or "European" not in missing

        # Query with entity not in chunks
        covered, missing = _check_entity_coverage(
            query="What is Japan doing about AI?",
            chunks=sample_chunks,
        )
        # Japan is not in the sample chunks
        assert not covered or "Japan" in missing


# ============================================================================
# Article Ingestor Tests
# ============================================================================


class TestArticleIngestor:
    """Tests for article ingestion."""

    def test_chunk_article(self, sample_articles):
        """Test article chunking."""
        from src.news_rag.core.article_ingestor import chunk_article

        article = sample_articles[0]
        chunks = chunk_article(article, "conv_123")

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.article_id == article.id
            assert chunk.conversation_id == "conv_123"
            assert chunk.url == article.url
            assert len(chunk.content) > 0

    def test_chunk_article_empty_content(self):
        """Test chunking with empty content."""
        from src.news_rag.core.article_ingestor import chunk_article

        article = Article(
            id="empty",
            title="Empty Article",
            url="https://example.com/empty",
            source="Test",
            content="",
        )

        chunks = chunk_article(article, "conv_123")
        assert len(chunks) == 0

    def test_get_article_ids_from_chunks(self):
        """Test extracting unique article IDs from chunks."""
        from src.news_rag.core.article_ingestor import get_article_ids_from_chunks

        chunks = [
            ArticleChunk(
                chunk_id="c1",
                article_id="a1",
                conversation_id="conv",
                content="test",
                chunk_index=0,
                url="",
                title="",
                source="",
            ),
            ArticleChunk(
                chunk_id="c2",
                article_id="a1",
                conversation_id="conv",
                content="test",
                chunk_index=1,
                url="",
                title="",
                source="",
            ),
            ArticleChunk(
                chunk_id="c3",
                article_id="a2",
                conversation_id="conv",
                content="test",
                chunk_index=0,
                url="",
                title="",
                source="",
            ),
        ]

        article_ids = get_article_ids_from_chunks(chunks)
        assert article_ids == ["a1", "a2"]


# ============================================================================
# Vector Retriever Tests
# ============================================================================


class TestVectorRetriever:
    """Tests for vector retrieval utilities."""

    def test_chunks_to_source_references(self, sample_chunks):
        """Test converting chunks to unique source references."""
        from src.news_rag.core.vector_retriever import chunks_to_source_references

        sources = chunks_to_source_references(sample_chunks)

        # Should deduplicate by article_id
        assert len(sources) == 2  # article_1 and article_2
        article_ids = [s.article_id for s in sources]
        assert "article_1" in article_ids
        assert "article_2" in article_ids

    def test_format_chunks_for_context(self, sample_chunks):
        """Test formatting chunks for LLM context."""
        from src.news_rag.core.vector_retriever import format_chunks_for_context

        context = format_chunks_for_context(sample_chunks)

        assert "[Source 1:" in context
        assert "[Source 2:" in context
        assert "European Union" in context
        assert "TechNews" in context

    def test_format_chunks_empty(self):
        """Test formatting empty chunks."""
        from src.news_rag.core.vector_retriever import format_chunks_for_context

        context = format_chunks_for_context([])
        assert "No relevant sources" in context

    def test_get_unique_article_count(self, sample_chunks):
        """Test counting unique articles."""
        from src.news_rag.core.vector_retriever import get_unique_article_count

        count = get_unique_article_count(sample_chunks)
        assert count == 2

    def test_get_average_similarity(self, sample_chunks):
        """Test calculating average similarity."""
        from src.news_rag.core.vector_retriever import get_average_similarity

        avg = get_average_similarity(sample_chunks)
        expected = (0.85 + 0.78 + 0.72) / 3
        assert abs(avg - expected) < 0.01

    def test_get_top_similarity(self, sample_chunks):
        """Test getting top similarity score."""
        from src.news_rag.core.vector_retriever import get_top_similarity

        top = get_top_similarity(sample_chunks)
        assert top == 0.85


# ============================================================================
# Answer Generator Tests
# ============================================================================


class TestAnswerGenerator:
    """Tests for answer generation utilities."""

    def test_map_sources_used_to_references(self, sample_chunks):
        """Test mapping source indices to references."""
        from src.news_rag.core.answer_generator import map_sources_used_to_references

        # Sources are 1-indexed in the answer
        sources_used = [1, 3]  # First and third chunks
        references = map_sources_used_to_references(sources_used, sample_chunks)

        # Should get unique articles
        assert len(references) == 2
        article_ids = [r.article_id for r in references]
        assert "article_1" in article_ids
        assert "article_2" in article_ids

    def test_map_sources_out_of_range(self, sample_chunks):
        """Test mapping with out-of-range indices."""
        from src.news_rag.core.answer_generator import map_sources_used_to_references

        sources_used = [1, 10, 100]  # 10 and 100 are out of range
        references = map_sources_used_to_references(sources_used, sample_chunks)

        # Should only get valid references
        assert len(references) == 1


# ============================================================================
# Integration-style Tests (with mocks)
# ============================================================================


class TestRAGGraphIntegration:
    """Integration tests for the RAG graph with mocked external calls."""

    @patch("src.news_rag.core.rag_graph.retrieve_articles")
    @patch("src.news_rag.core.rag_graph.ingest_articles")
    @patch("src.news_rag.core.rag_graph.retrieve_relevant_chunks")
    @patch("src.news_rag.core.rag_graph.generate_summary_answer")
    @patch("src.news_rag.core.vector_store.get_chunks_by_conversation")
    def test_initial_query_flow(
        self,
        mock_get_chunks,
        mock_generate_summary,
        mock_retrieve_chunks,
        mock_ingest,
        mock_retrieve_articles,
        sample_articles,
        sample_chunks,
    ):
        """Test the initial query flow through the RAG graph."""
        from src.news_rag.core.rag_graph import run_news_query

        # Setup mocks
        mock_get_chunks.return_value = []  # No existing chunks = initial query
        mock_retrieve_articles.return_value = sample_articles
        mock_ingest.return_value = (3, 6)  # 3 articles, 6 chunks
        mock_retrieve_chunks.return_value = sample_chunks
        mock_generate_summary.return_value = {
            "answer": "Summary of AI regulations...",
            "sources_used": [1, 2],
            "confidence": "high",
            "missing_info": None,
        }

        # Run the query
        response = run_news_query(
            user_id="test_user",
            conversation_id=None,
            message="What are the latest AI regulations?",
        )

        # Verify response
        assert response.answer_type == "summary"
        assert "AI regulations" in response.answer_text or len(response.answer_text) > 0
        assert response.conversation_id is not None

    @patch("src.news_rag.core.rag_graph.retrieve_relevant_chunks")
    @patch("src.news_rag.core.rag_graph.check_sufficiency")
    @patch("src.news_rag.core.rag_graph.generate_answer")
    @patch("src.news_rag.core.vector_store.get_chunks_by_conversation")
    def test_followup_query_sufficient(
        self,
        mock_get_chunks,
        mock_generate_answer,
        mock_check_sufficiency,
        mock_retrieve_chunks,
        sample_chunks,
    ):
        """Test follow-up query when stored sources are sufficient."""
        from src.news_rag.core.rag_graph import run_news_query

        # Setup mocks
        mock_get_chunks.return_value = sample_chunks  # Has existing chunks = follow-up
        mock_retrieve_chunks.return_value = sample_chunks
        mock_check_sufficiency.return_value = (True, "Sufficient")
        mock_generate_answer.return_value = {
            "answer": "The EU proposed the AI Act...",
            "sources_used": [1],
            "confidence": "high",
            "missing_info": None,
        }

        # Run follow-up query
        response = run_news_query(
            user_id="test_user",
            conversation_id="existing_conv_123",
            message="What specific regulations were proposed?",
        )

        # Verify response
        assert response.answer_type == "followup_answer"
        assert response.conversation_id == "existing_conv_123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
