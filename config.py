"""Configuration module for NBA RAG pipeline.

Loads configuration parameters from environment variables (.env) and defines
typed paths, model identifiers, and API key accessors with clear fail-fast
validation and deprecation warnings for legacy key names.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Ensure .env is loaded
load_dotenv()

logger = logging.getLogger(__name__)

# Embedding constants
EMBED_MODEL: str = "gemini-embedding-001"
EMBED_DIM: int = 1536
DEFAULT_COLLECTION: str = "nba_guide__gemini-001__d1536"

# Default model & paths
DEFAULT_LLM_MODEL: str = "gemini-2.5-flash"
LLM_MODEL: str = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
DB_PATH: Path = Path(os.getenv("DB_PATH", "./nba_stats.db"))


def get_embed_api_key(required: bool = True) -> str:
    """Retrieves the Gemini embedding API key with deprecation fallback.

    Used strictly for embedding operations (ingestion and retrieval queries).
    Checks GEMINI_EMBED_API_KEY first. If missing, checks legacy GEMINI_API_KEY
    or GOOGLE_API_KEY and emits a deprecation warning.

    Args:
        required: Whether to raise a ValueError if no key is found.

    Returns:
        The API key string, or empty string if not required and missing.

    Raises:
        ValueError: If required is True and the key is missing.
    """
    key = os.getenv("GEMINI_EMBED_API_KEY")
    if key:
        return key

    fallback = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if fallback:
        legacy_name = "GEMINI_API_KEY" if os.getenv("GEMINI_API_KEY") else "GOOGLE_API_KEY"
        warnings.warn(
            f"Using legacy environment variable '{legacy_name}' for embeddings. "
            "Please migrate to 'GEMINI_EMBED_API_KEY'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return fallback

    if required:
        raise ValueError(
            "Missing required environment variable 'GEMINI_EMBED_API_KEY'. "
            "Please set GEMINI_EMBED_API_KEY in your environment or .env file."
        )
    return ""


def get_llm_api_key(required: bool = True) -> str:
    """Retrieves the Gemini chat LLM API key.

    Used strictly for chat LLM interactions (LangChain agent).

    Args:
        required: Whether to raise a ValueError if no key is found.

    Returns:
        The API key string, or empty string if not required and missing.

    Raises:
        ValueError: If required is True and the key is missing.
    """
    key = os.getenv("GEMINI_LLM_API_KEY")
    if key:
        return key

    if required:
        raise ValueError(
            "Missing required environment variable 'GEMINI_LLM_API_KEY'. "
            "Please set GEMINI_LLM_API_KEY in your environment or .env file."
        )
    return ""


@dataclass(frozen=True)
class Config:
    """Typed configuration container for NBA RAG pipeline."""

    gemini_embed_api_key: str
    gemini_llm_api_key: str
    llm_model: str = LLM_MODEL
    data_dir: Path = DATA_DIR
    db_path: Path = DB_PATH
    collection_name: str = DEFAULT_COLLECTION

    @classmethod
    def from_env(cls) -> Config:
        """Instantiates Config by reading all environment variables.

        Returns:
            Configured Config instance.

        Raises:
            ValueError: If any required environment variable is missing.
        """
        return cls(
            gemini_embed_api_key=get_embed_api_key(required=True),
            gemini_llm_api_key=get_llm_api_key(required=True),
            llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
            data_dir=Path(os.getenv("DATA_DIR", "./data")),
            db_path=Path(os.getenv("DB_PATH", "./nba_stats.db")),
            collection_name=DEFAULT_COLLECTION,
        )


def __getattr__(name: str) -> Any:
    """Dynamic attribute accessor for module-level constants.

    Ensures fail-fast validation when GEMINI_EMBED_API_KEY or
    GEMINI_LLM_API_KEY are accessed directly.
    """
    if name == "GEMINI_EMBED_API_KEY":
        return get_embed_api_key(required=True)
    if name == "GEMINI_LLM_API_KEY":
        return get_llm_api_key(required=True)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
