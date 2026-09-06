"""
InsightAI - Application settings.

Reads configuration from environment variables and an optional ``.env`` file,
and exposes them as a typed Pydantic model. Every setting has a sensible
default so the application runs out-of-the-box even without a ``.env`` file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv is optional
    load_dotenv = None

from pydantic import BaseModel, Field

# Project root = folder containing this file's parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseModel):
    """Typed application configuration."""

    # ------------------------------------------------------------------ LLM
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        description="Base URL of the local Ollama server.",
    )
    ollama_model: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        description="Default Ollama model name.",
    )
    ollama_temperature: float = Field(
        default=float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
    )
    ollama_max_tokens: int = Field(
        default=int(os.getenv("OLLAMA_MAX_TOKENS", "1200"))
    )
    ollama_timeout: int = Field(
        default=int(os.getenv("OLLAMA_TIMEOUT", "90"))
    )

    # ------------------------------------------------------------------ RAG
    embedding_model: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        description="Sentence-transformer model used for RAG embeddings.",
    )
    vector_db_path: str = Field(
        default_factory=lambda: os.getenv("VECTOR_DB_PATH", "rag/vectorstore"),
        description="Path where the vector index is persisted.",
    )
    rag_top_k: int = Field(
        default=int(os.getenv("RAG_TOP_K", "5")),
        description="Number of chunks to retrieve per query.",
    )

    # ------------------------------------------------------------------ App
    max_upload_size_mb: int = Field(
        default=int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))
    )
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper(),
    )
    sample_sizes: List[float] = Field(
        default_factory=lambda: [1.0, 0.5, 0.25, 0.10],
        description="Allowed sampling fractions shown in the UI.",
    )

    # ------------------------------------------------------------------ Paths
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    exports_dir: Path = PROJECT_ROOT / "data" / "exports"
    knowledge_base_dir: Path = PROJECT_ROOT / "rag" / "knowledge_base"

    # Thresholds used by the heuristics & outlier classifier.
    missing_warn_threshold: float = 0.05   # % of missing values -> warn
    low_cardinality_threshold: int = 50    # below this a column is "categorical"
    extreme_cardinality_threshold: int = 200

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.raw_dir, self.processed_dir, self.exports_dir):
            path.mkdir(parents=True, exist_ok=True)

    def model_post_init(self, *args, **kwargs) -> None:
        self.ensure_dirs()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
