from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "startup_postmortems"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5

    use_hybrid_search: bool = True
    sparse_model: str = "Qdrant/bm25"

    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 3

    model_config = {"env_file": ".env"}


settings = Settings()
