"""Shared pytest setup: tests never read the developer's .env or real API keys."""

from __future__ import annotations

from typing import Any

import dotenv
import pytest

# Every environment variable the project reads.
PROJECT_ENV_VARS: tuple[str, ...] = (
    "GEMINI_EMBED_API_KEY",
    "GEMINI_LLM_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "DATA_DIR",
    "DB_PATH",
)

_session_patch = pytest.MonkeyPatch()


def _no_dotenv(*args: Any, **kwargs: Any) -> bool:
    return False


def pytest_configure(config: pytest.Config) -> None:
    # Runs before test modules are imported, so `from dotenv import load_dotenv` in config.py,
    # embed_data.py and query.py binds this no-op and their import-time load_dotenv() reads nothing.
    _session_patch.setattr(dotenv, "load_dotenv", _no_dotenv)


def pytest_unconfigure(config: pytest.Config) -> None:
    _session_patch.undo()


@pytest.fixture(autouse=True)
def _clean_project_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Removes project variables (e.g. exported in the shell) so each test sets exactly what it needs."""
    for name in PROJECT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
