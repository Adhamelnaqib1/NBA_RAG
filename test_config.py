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

    @pytest.mark.parametrize(("provider", "var"), [(None, "NVIDIA_API_KEY"), ("nvidia", "NVIDIA_API_KEY"),
                                                   ("gemini", "GEMINI_LLM_API_KEY"), (" Gemini ", "GEMINI_LLM_API_KEY")])
    def test_missing_llm_key_names_the_provider_variable(
        self, monkeypatch: pytest.MonkeyPatch, provider: str | None, var: str
    ) -> None:
        if provider is not None:
            monkeypatch.setenv("LLM_PROVIDER", provider)
        # The other provider's key is set and must not be used.
        other = "GEMINI_LLM_API_KEY" if var == "NVIDIA_API_KEY" else "NVIDIA_API_KEY"
        monkeypatch.setenv(other, "other-provider-secret")
        with pytest.raises(ValueError) as exc_info:
            config.get_llm_api_key(required=True)
        msg = str(exc_info.value)
        assert f"Missing required environment variable '{var}'" in msg
        assert "secret" not in msg

    @pytest.mark.parametrize(("provider", "var"), [("nvidia", "NVIDIA_API_KEY"), ("gemini", "GEMINI_LLM_API_KEY")])
    def test_llm_key_follows_provider(self, monkeypatch: pytest.MonkeyPatch, provider: str, var: str) -> None:
        monkeypatch.setenv("LLM_PROVIDER", provider)
        monkeypatch.setenv(var, f"{provider}-key")
        assert config.get_llm_api_key() == f"{provider}-key"

    def test_llm_provider_default_and_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert config.get_llm_provider() == "nvidia"
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        with pytest.raises(ValueError, match="LLM_PROVIDER must be one of nvidia, gemini; got 'openai'"):
            config.get_llm_provider()
        with pytest.raises(ValueError, match="LLM_PROVIDER"):
            config.get_llm_api_key()

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
        monkeypatch.setenv("NVIDIA_API_KEY", "nim")
        monkeypatch.setenv("GEMINI_LLM_API_KEY", "free")
        assert config.get_embed_api_key() == "paid"
        assert config.get_llm_api_key() == "nim"
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        assert config.get_llm_api_key() == "free"
        assert config.get_embed_api_key() == "paid"

    @pytest.mark.parametrize(("provider", "model"), [(None, "nvidia/nemotron-3.5-super-vl-120b-a12b"),
                                                     ("gemini", "gemini-2.5-flash")])
    def test_llm_model_default_follows_provider(
        self, monkeypatch: pytest.MonkeyPatch, provider: str | None, model: str
    ) -> None:
        if provider is not None:
            monkeypatch.setenv("LLM_PROVIDER", provider)
        assert config.get_llm_model() == model
        assert config.LLM_MODEL == model
        cfg = config.Config(gemini_embed_api_key="k1", llm_api_key="k2")
        assert (cfg.llm_provider, cfg.llm_model) == (provider or "nvidia", model)

    def test_llm_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_MODEL", "nvidia/some-new-model")
        monkeypatch.setenv("GEMINI_EMBED_API_KEY", "k1")
        monkeypatch.setenv("NVIDIA_API_KEY", "k2")
        loaded = config.Config.from_env()
        assert (loaded.llm_provider, loaded.llm_model, loaded.llm_api_key) == ("nvidia", "nvidia/some-new-model", "k2")
        assert config.LLM_MODEL == "nvidia/some-new-model"

    @pytest.mark.parametrize(
        ("var", "getter", "raw", "expected"),
        [
            ("LLM_MAX_TOKENS", "get_llm_max_tokens", None, None),
            ("LLM_MAX_TOKENS", "get_llm_max_tokens", "  ", None),
            ("LLM_MAX_TOKENS", "get_llm_max_tokens", " 8192 ", 8192),
            ("LLM_TIMEOUT_SECONDS", "get_llm_timeout_seconds", None, 120.0),
            ("LLM_TIMEOUT_SECONDS", "get_llm_timeout_seconds", "45.5", 45.5),
        ],
    )
    def test_llm_limits(
        self, monkeypatch: pytest.MonkeyPatch, var: str, getter: str, raw: str | None, expected: float | None
    ) -> None:
        if raw is not None:
            monkeypatch.setenv(var, raw)
        assert getattr(config, getter)() == expected

    @pytest.mark.parametrize(
        ("var", "getter", "raw"),
        [("LLM_MAX_TOKENS", "get_llm_max_tokens", "lots"), ("LLM_MAX_TOKENS", "get_llm_max_tokens", "0"),
         ("LLM_MAX_TOKENS", "get_llm_max_tokens", "1.5"), ("LLM_TIMEOUT_SECONDS", "get_llm_timeout_seconds", "-1")],
    )
    def test_llm_limits_invalid(self, monkeypatch: pytest.MonkeyPatch, var: str, getter: str, raw: str) -> None:
        monkeypatch.setenv(var, raw)
        with pytest.raises(ValueError, match=f"{var} must be a positive"):
            getattr(config, getter)()

    @pytest.mark.parametrize(("raw", "expected"), [(None, None), ("", None), ("  ", None), ("0", 0.0), (" 0.7 ", 0.7)])
    def test_llm_temperature(self, monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: float | None) -> None:
        if raw is not None:
            monkeypatch.setenv("LLM_TEMPERATURE", raw)
        assert config.get_llm_temperature() == expected

    def test_llm_temperature_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_TEMPERATURE", "hot")
        with pytest.raises(ValueError, match="LLM_TEMPERATURE must be a number"):
            config.get_llm_temperature()

    def test_config_paths_and_collection(self) -> None:
        """Tests paths and default collection resolution."""
        cfg = config.Config(
            gemini_embed_api_key="k1",
            llm_api_key="k2",
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
        with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
            _ = config.LLM_API_KEY


class TestEnvIsolation:
    """Proves the test run is independent of any .env file."""

    def test_config_load_dotenv_is_disabled(self) -> None:
        # The real function lives in dotenv.main; conftest swaps the dotenv.load_dotenv name that config imports.
        assert config.load_dotenv is not dotenv.main.load_dotenv

    @pytest.mark.usefixtures("reloaded_config")
    def test_reimport_keeps_defaults(self) -> None:
        assert "LLM_MODEL" not in os.environ
        assert config.LLM_MODEL == config.DEFAULT_LLM_MODELS["nvidia"]
        assert config.DB_PATH == Path("./nba_stats.db")
        assert config.get_llm_api_key(required=False) == ""
