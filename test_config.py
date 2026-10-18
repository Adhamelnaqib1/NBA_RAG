"""Unit tests for config.py and API key handling."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

import config


class TestConfig:
    """Tests for configuration parsing, fail-fast validation, and key separation."""

    def test_missing_embed_key_fails_fast(self) -> None:
        """Asserts missing embed key raises ValueError naming variable without secret leaking."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                config.get_embed_api_key(required=True)
            msg = str(exc_info.value)
            assert "GEMINI_EMBED_API_KEY" in msg
            assert "Missing required environment variable" in msg

    def test_missing_llm_key_fails_fast(self) -> None:
        """Asserts missing LLM key raises ValueError naming variable without secret leaking."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                config.get_llm_api_key(required=True)
            msg = str(exc_info.value)
            assert "GEMINI_LLM_API_KEY" in msg
            assert "Missing required environment variable" in msg

    def test_embed_key_legacy_fallback_warning(self) -> None:
        """Asserts fallback to legacy keys emits DeprecationWarning."""
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "legacy_key_123"}, clear=True),
            pytest.deprecated_call(match="Using legacy environment variable 'GEMINI_API_KEY'"),
        ):
            key = config.get_embed_api_key(required=True)
            assert key == "legacy_key_123"

        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "legacy_key_456"}, clear=True),
            pytest.deprecated_call(match="Using legacy environment variable 'GOOGLE_API_KEY'"),
        ):
            key = config.get_embed_api_key(required=True)
            assert key == "legacy_key_456"

    def test_embed_key_preferred_over_legacy(self) -> None:
        """Asserts GEMINI_EMBED_API_KEY is preferred and does not emit warnings."""
        env = {
            "GEMINI_EMBED_API_KEY": "embed_key_paid",
            "GEMINI_API_KEY": "legacy_key",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            warnings.catch_warnings(record=True) as recorded,
        ):
            warnings.simplefilter("always")
            key = config.get_embed_api_key(required=True)
            assert key == "embed_key_paid"
            assert len(recorded) == 0

    def test_llm_model_default_and_override(self) -> None:
        """Tests that LLM_MODEL respects environment override or sensible default."""
        with patch.dict(os.environ, {}, clear=True):
            # Default model is gemini-2.5-flash
            cfg = config.Config(
                gemini_embed_api_key="k1",
                gemini_llm_api_key="k2",
            )
            assert cfg.llm_model == "gemini-2.5-flash"

        with (
            patch.dict(os.environ, {"LLM_MODEL": "gemini-2.0-flash"}, clear=True),
            patch.dict(
                os.environ,
                {"GEMINI_EMBED_API_KEY": "k1", "GEMINI_LLM_API_KEY": "k2"},
            ),
        ):
            loaded = config.Config.from_env()
            assert loaded.llm_model == "gemini-2.0-flash"

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
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GEMINI_EMBED_API_KEY"):
                _ = config.GEMINI_EMBED_API_KEY
            with pytest.raises(ValueError, match="GEMINI_LLM_API_KEY"):
                _ = config.GEMINI_LLM_API_KEY
