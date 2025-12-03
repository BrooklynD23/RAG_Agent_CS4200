from src.news_rag.core.router import classify_query


def test_classify_query_news_keyword() -> None:
    assert classify_query("latest news today") == "news"
