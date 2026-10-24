"""Configuration module for NBA RAG pipeline.

Loads configuration parameters from environment variables (.env) and defines
typed paths, model identifiers, and API key accessors with clear fail-fast
validation and deprecation warnings for legacy key names.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
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

# Chat LLM provider, model and limits (Gemini embeddings use the paid key above; see EMBED_BACKENDS below)
LLM_PROVIDERS: tuple[str, ...] = ("nvidia", "gemini")
DEFAULT_LLM_PROVIDER: str = "nvidia"
DEFAULT_LLM_MODELS: dict[str, str] = {
    "nvidia": "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "gemini": "gemini-2.5-flash",
}
LLM_API_KEY_VARS: dict[str, str] = {"nvidia": "NVIDIA_API_KEY", "gemini": "GEMINI_LLM_API_KEY"}
DEFAULT_LLM_TIMEOUT_SECONDS: float = 120.0
DEFAULT_APP_MAX_QUESTIONS: int = 30
DEFAULT_SEARCH_BUDGET: int = 1
DEFAULT_SQL_BUDGET: int = 3
DEFAULT_TOTAL_TOOL_BUDGET: int = 4
DEFAULT_SEARCH_K: int = 8
MAX_SEARCH_K: int = 10

DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
DB_PATH: Path = Path(os.getenv("DB_PATH", "./nba_stats.db"))

# Embedding backends. gemini: gemini-embedding-001 on the paid key, store in DATA_DIR (the default).
# bge: BAAI/bge-base-en-v1.5 run locally with sentence-transformers (no key), in its own store BGE_DATA_DIR so the
# Gemini store is never touched. Ingestion and queries both go through embed_data.embed_texts with the same spec.
EMBED_BACKENDS: tuple[str, ...] = ("gemini", "bge")
DEFAULT_EMBED_BACKEND: str = "gemini"
BGE_MODEL: str = "BAAI/bge-base-en-v1.5"
BGE_DIM: int = 768
BGE_COLLECTION: str = "nba_guide__bge-base-en-v1.5__d768"
LOCAL_EMBED_MODELS: frozenset[str] = frozenset({BGE_MODEL})
EMBED_DEVICES: tuple[str, ...] = ("auto", "mps", "cuda", "cpu")


@dataclass(frozen=True)
class EmbedSpec:
    """Everything that must match between ingestion and query time for one embedding backend."""

    backend: str
    model: str
    dim: int
    collection: str
    data_dir: Path

    @property
    def local(self) -> bool:
        return self.model in LOCAL_EMBED_MODELS


def get_embed_backend() -> str:
    """EMBED_BACKEND, lowercased: "gemini" (default) or "bge".

    Raises:
        ValueError: For any other value.
    """
    backend = os.getenv("EMBED_BACKEND", "").strip().lower() or DEFAULT_EMBED_BACKEND
    if backend not in EMBED_BACKENDS:
        raise ValueError(f"EMBED_BACKEND must be one of {', '.join(EMBED_BACKENDS)}; got {backend!r}.")
    return backend


def get_embed_spec(backend: str | None = None) -> EmbedSpec:
    """The spec for `backend` (default: EMBED_BACKEND).

    Raises:
        ValueError: If the backend is unknown.
    """
    backend = backend or get_embed_backend()
    if backend == "gemini":
        return EmbedSpec("gemini", EMBED_MODEL, EMBED_DIM, DEFAULT_COLLECTION, DATA_DIR)
    if backend == "bge":
        return EmbedSpec("bge", BGE_MODEL, BGE_DIM, BGE_COLLECTION, Path(os.getenv("BGE_DATA_DIR", "./data_bge")))
    raise ValueError(f"Embedding backend must be one of {', '.join(EMBED_BACKENDS)}; got {backend!r}.")


def get_embed_device() -> str:
    """EMBED_DEVICE for local embedding models: auto (default: mps, then cuda, then cpu), mps, cuda or cpu.

    Raises:
        ValueError: For any other value.
    """
    device = os.getenv("EMBED_DEVICE", "").strip().lower() or "auto"
    if device not in EMBED_DEVICES:
        raise ValueError(f"EMBED_DEVICE must be one of {', '.join(EMBED_DEVICES)}; got {device!r}.")
    return device


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


def get_llm_provider() -> str:
    """LLM_PROVIDER, lowercased: "nvidia" (default) or "gemini".

    Raises:
        ValueError: For any other value.
    """
    provider = os.getenv("LLM_PROVIDER", "").strip().lower() or DEFAULT_LLM_PROVIDER
    if provider not in LLM_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {', '.join(LLM_PROVIDERS)}; got {provider!r}.")
    return provider


def get_llm_model() -> str:
    """LLM_MODEL, or the default model for the current provider when unset."""
    return os.getenv("LLM_MODEL", "").strip() or DEFAULT_LLM_MODELS[get_llm_provider()]


def get_llm_api_key(required: bool = True) -> str:
    """Retrieves the chat LLM API key for the current provider.

    nvidia -> NVIDIA_API_KEY, gemini -> GEMINI_LLM_API_KEY (free tier). Never the paid embedding key.

    Args:
        required: Whether to raise a ValueError if no key is found.

    Returns:
        The API key string, or empty string if not required and missing.

    Raises:
        ValueError: If required is True and the key is missing (names the variable, never a value),
            or if LLM_PROVIDER is invalid.
    """
    var = LLM_API_KEY_VARS[get_llm_provider()]
    key = os.getenv(var)
    if key:
        return key

    if required:
        raise ValueError(
            f"Missing required environment variable '{var}'. "
            f"Please set {var} in your environment or .env file."
        )
    return ""


def _positive_number(var: str, default: float | None, cast: type[int | float]) -> Any:
    raw = os.getenv(var, "").strip()
    if not raw:
        return None if default is None else cast(default)
    try:
        value = cast(raw)
    except ValueError as e:
        raise ValueError(f"{var} must be a positive {cast.__name__}, got {raw!r}.") from e
    if value <= 0:
        raise ValueError(f"{var} must be a positive {cast.__name__}, got {raw!r}.")
    return value


def get_llm_max_tokens() -> int | None:
    """LLM_MAX_TOKENS: output cap per model call (reasoning tokens may count toward it), or None when unset
    so the chat library's default applies."""
    value = _positive_number("LLM_MAX_TOKENS", None, int)
    return None if value is None else int(value)


def get_search_budget() -> int:
    """SEARCH_BUDGET (default 1): search_guide calls allowed per question."""
    return int(_positive_number("SEARCH_BUDGET", DEFAULT_SEARCH_BUDGET, int))


def get_sql_budget() -> int:
    """SQL_BUDGET (default 3): run_sql calls allowed per question."""
    return int(_positive_number("SQL_BUDGET", DEFAULT_SQL_BUDGET, int))


def get_total_tool_budget() -> int:
    """TOTAL_TOOL_BUDGET (default 4): tool call attempts (allowed or refused) per question."""
    return int(_positive_number("TOTAL_TOOL_BUDGET", DEFAULT_TOTAL_TOOL_BUDGET, int))


def get_search_k() -> int:
    """SEARCH_K (default 8, 1-10): passages search_guide returns when the model does not pass k."""
    k = int(_positive_number("SEARCH_K", DEFAULT_SEARCH_K, int))
    if k > MAX_SEARCH_K:
        raise ValueError(f"SEARCH_K must be an integer from 1 to {MAX_SEARCH_K}, got {k}.")
    return k


ACRONYM_MODES: tuple[str, ...] = ("both", "fallback", "off")
DEFAULT_ACRONYM_MODE: str = "fallback"


def get_acronym_mode() -> str:
    """ACRONYM_MODE: how search_guide uses acronym expansion (acronyms.yaml).

    fallback (default): search the expanded query only when no passage of the original search mentions the
    matched term; both: search the original and the expanded query together and merge; off: no expansion.

    Raises:
        ValueError: For any other value.
    """
    mode = os.getenv("ACRONYM_MODE", "").strip().lower() or DEFAULT_ACRONYM_MODE
    if mode not in ACRONYM_MODES:
        raise ValueError(f"ACRONYM_MODE must be one of {', '.join(ACRONYM_MODES)}; got {mode!r}.")
    return mode


def get_offtopic_guard() -> bool:
    """OFFTOPIC_GUARD: "1" (default) answers clearly non-NBA questions without calling the model; "0" disables."""
    raw = os.getenv("OFFTOPIC_GUARD", "").strip() or "1"
    if raw not in ("0", "1"):
        raise ValueError(f"OFFTOPIC_GUARD must be 0 or 1, got {raw!r}.")
    return raw == "1"


def get_app_max_questions() -> int:
    """APP_MAX_QUESTIONS (default 30): questions one Streamlit session may ask, to protect the LLM key."""
    return int(_positive_number("APP_MAX_QUESTIONS", DEFAULT_APP_MAX_QUESTIONS, int))


def get_llm_timeout_seconds() -> float:
    """LLM_TIMEOUT_SECONDS (default 120): HTTP timeout for one model call."""
    return float(_positive_number("LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS, float))


@dataclass(frozen=True)
class Config:
    """Typed configuration container for NBA RAG pipeline."""

    gemini_embed_api_key: str
    llm_api_key: str
    llm_provider: str = field(default_factory=get_llm_provider)
    llm_model: str = field(default_factory=get_llm_model)
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
            llm_api_key=get_llm_api_key(required=True),
            llm_provider=get_llm_provider(),
            llm_model=get_llm_model(),
            data_dir=Path(os.getenv("DATA_DIR", "./data")),
            db_path=Path(os.getenv("DB_PATH", "./nba_stats.db")),
            collection_name=DEFAULT_COLLECTION,
        )


def get_llm_temperature() -> float | None:
    """Returns LLM_TEMPERATURE as a float, or None when unset (the model's default is then used).

    Raises:
        ValueError: If LLM_TEMPERATURE is set but is not a number.
    """
    raw = os.getenv("LLM_TEMPERATURE", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"LLM_TEMPERATURE must be a number, got {raw!r}.") from e


def __getattr__(name: str) -> Any:
    """Dynamic attribute accessor for module-level constants.

    Ensures fail-fast validation when GEMINI_EMBED_API_KEY or the LLM key are accessed directly,
    and reads LLM_MODEL at access time so it follows LLM_PROVIDER.
    """
    if name == "GEMINI_EMBED_API_KEY":
        return get_embed_api_key(required=True)
    if name == "LLM_API_KEY":
        return get_llm_api_key(required=True)
    if name == "LLM_MODEL":
        return get_llm_model()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
