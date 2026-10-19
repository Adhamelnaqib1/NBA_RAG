"""Unit tests for config.py and API key handling.

conftest.py removes every project variable before each test and disables load_dotenv,
so these tests never see the developer's .env.
"""

from __future__ import annotations

import importlib
import os
import warnings
from collections.abc import Iterator
from pathlib import Path

import dotenv.main
import pytest

import config


@pytest.fixture
def reloaded_config() -> Iterator[None]:
    """Re-runs config's import-time code (including its load_dotenv() call), then restores it."""
    importlib.reload(config)
    yield
    importlib.reload(config)


class TestConfig:
    """Tests for configuration parsing, fail-fast validation, and key separation."""

    def test_missing_embed_key_fails_fast(self) -> None:
        """Asserts missing embed key raises ValueError naming variable without secret leaking."""
        with pytest.raises(ValueError) as exc_info:
            config.get_embed_api_key(required=True)
        msg = str(exc_info.value)
        assert "GEMINI_EMBED_API_KEY" in msg
        assert "Missing required environment variable" in msg

    def test_missing_llm_key_fails_fast(self) -> None:
        """Asserts missing LLM key raises ValueError naming variable without secret leaking."""
        with pytest.raises(ValueError) as exc_info:
            config.get_llm_api_key(required=True)
        msg = str(exc_info.value)
        assert "GEMINI_LLM_API_KEY" in msg
        assert "Missing required environment variable" in msg

    def test_optional_keys_return_empty(self) -> None:
        assert config.get_embed_api_key(required=False) == ""
        assert config.get_llm_api_key(required=False) == ""

    @pytest.mark.parametrize("legacy_name", ["GEMINI_API_KEY", "GOOGLE_API_KEY"])
    def test_embed_key_legacy_fallback_warning(self, legacy_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Asserts fallback to legacy keys emits DeprecationWarning."""
        monkeypatch.setenv(legacy_name, "legacy_key_123")
        with pytest.deprecated_call(match=f"Using legacy environment variable '{legacy_name}'"):
            assert config.get_embed_api_key(required=True) == "legacy_key_123"

    def test_embed_key_preferred_over_legacy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Asserts GEMINI_EMBED_API_KEY is preferred and does not emit warnings."""
        monkeypatch.setenv("GEMINI_EMBED_API_KEY", "embed_key_paid")
        monkeypatch.setenv("GEMINI_API_KEY", "legacy_key")
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            assert config.get_embed_api_key(required=True) == "embed_key_paid"
        assert len(recorded) == 0

    def test_embed_and_llm_keys_are_separate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_EMBED_API_KEY", "paid")
        monkeypatch.setenv("GEMINI_LLM_API_KEY", "free")
        assert config.get_embed_api_key() == "paid"
        assert config.get_llm_api_key() == "free"

    def test_llm_model_default(self) -> None:
        cfg = config.Config(gemini_embed_api_key="k1", gemini_llm_api_key="k2")
        assert cfg.llm_model == config.DEFAULT_LLM_MODEL == "gemini-2.5-flash"

    def test_llm_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_MODEL", "gemini-2.0-flash")
        monkeypatch.setenv("GEMINI_EMBED_API_KEY", "k1")
        monkeypatch.setenv("GEMINI_LLM_API_KEY", "k2")
        assert config.Config.from_env().llm_model == "gemini-2.0-flash"
        assert config.Config(gemini_embed_api_key="k1", gemini_llm_api_key="k2").llm_model == "gemini-2.0-flash"

    def test_config_paths_and_collection(self) -> None:
        """Tests paths and default collection resolution."""
        cfg = config.Config(
            gemini_embed_api_key="k1",
            gemini_llm_api_key="k2",
            data_dir=Path("./test_data"),
            db_path=Path("./test.db"),
        )
        assert cfg.data_dir == Path("./test_data")
        assert cfg.db_path == Path("./test.db")
        assert cfg.collection_name == config.DEFAULT_COLLECTION

    def test_module_getattr_fail_fast(self) -> None:
        """Tests module attribute access fails fast if keys are missing."""
        with pytest.raises(ValueError, match="GEMINI_EMBED_API_KEY"):
            _ = config.GEMINI_EMBED_API_KEY
        with pytest.raises(ValueError, match="GEMINI_LLM_API_KEY"):
            _ = config.GEMINI_LLM_API_KEY


class TestEnvIsolation:
    """Proves the test run is independent of any .env file."""

    def test_config_load_dotenv_is_disabled(self) -> None:
        # The real function lives in dotenv.main; conftest swaps the dotenv.load_dotenv name that config imports.
        assert config.load_dotenv is not dotenv.main.load_dotenv

    @pytest.mark.usefixtures("reloaded_config")
    def test_reimport_keeps_defaults(self) -> None:
        assert "LLM_MODEL" not in os.environ
        assert config.LLM_MODEL == config.DEFAULT_LLM_MODEL
        assert config.DB_PATH == Path("./nba_stats.db")
        assert config.get_llm_api_key(required=False) == ""
