from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    openai_api_key: str | None = None
    google_api_key: str | None = None
    tavily_api_key: str | None = None
    gnews_api_key: str | None = None

    news_rag_model_name: str = "gemini-1.5-flash"
    google_embedding_model: str = "models/text-embedding-004"
    google_chat_model: str = "gemini-1.5-flash"
    max_articles: int = 10
    chunk_size: int = 1000
    chunk_overlap: int = 150

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
