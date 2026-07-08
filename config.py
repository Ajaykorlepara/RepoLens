"""
Central configuration for the GitHub RAG Agent backend.
All values can be overridden via environment variables or a .env file.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM (Groq) ---
    groq_api_key: str = "YOUR_GROQ_API_KEY"
    groq_model: str = "openai/gpt-oss-120b"

    # --- Optional: for cloning private repos ---
    github_token: str = ""

    # --- Embeddings (runs locally, no API key needed) ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Storage paths ---
    chroma_persist_dir: str = "../data/chroma_db"
    clone_dir: str = "../data/repos"

    # --- Chunking / retrieval tuning ---
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 5
    max_file_size_kb: int = 500


settings = Settings()

# Make sure our storage folders exist on startup.
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
Path(settings.clone_dir).mkdir(parents=True, exist_ok=True)
