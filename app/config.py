from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "startup_postmortems"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5

    model_config = {"env_file": ".env"}


settings = Settings()
