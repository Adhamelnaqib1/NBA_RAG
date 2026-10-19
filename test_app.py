"""Offline tests for app.py with streamlit.testing.v1.AppTest and a mocked agent (no LLM or embedding calls)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.errors import AppTestError

import agent
from agent import AgentResult, ToolStep

APP = Path(__file__).parent / "app.py"
MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"


def full_result(question: str) -> AgentResult:
    return AgentResult(
        answer=f"Answer to: {question} (PDF p. 180).",
        steps=[
            ToolStep("run_sql", {"sql": "SELECT winner_team FROM nba_finals WHERE year = 2026"}, "| New York |", [180]),
            ToolStep("search_guide", {"query": "2026 Finals MVP"}, "--- PDF p. 102 ---", [102]),
        ],
        pages=[180, 999],
        verified_pages=[180],
        unverified_pages=[999],
        seconds=7.84,
        model_calls=3,
        input_tokens=8300,
        output_tokens=400,
    )


class FakeAgent:
    """Records calls; `reply` decides what ask() returns or raises."""

    def __init__(self, reply: Callable[[str], AgentResult] = full_result) -> None:
        self.reply = reply
        self.builds = 0
        self.calls: list[tuple[str, list[tuple[str, str]]]] = []

    def build(self) -> object:
        self.builds += 1
        return object()

    def ask(self, graph: object, question: str, history: list[tuple[str, str]] | None = None) -> AgentResult:
        assert st.session_state.pending == question  # answered in the run where the input is disabled
        self.calls.append((question, list(history or [])))
        return self.reply(question)


@pytest.fixture(autouse=True)
def clean_cache() -> Iterator[None]:
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeAgent:
    fake_agent = FakeAgent()
    monkeypatch.setattr(agent, "build_agent", fake_agent.build)
    monkeypatch.setattr(agent, "ask", fake_agent.ask)
    return fake_agent


def run_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def ask_in(at: AppTest, question: str) -> AppTest:
    at.chat_input[0].set_value(question).run()
    assert not at.exception
    return at


def texts(elements: Any) -> list[str]:
    return [e.value for e in elements]


class TestRender:
    def test_renders_title_input_sidebar_and_model(self, fake: FakeAgent) -> None:
        at = run_app()
        assert at.title[0].value == "NBA Guide Q&A"
        assert at.chat_input[0].placeholder == "Ask about the 2026-27 NBA Guide..."
        assert not at.chat_input[0].disabled
        sidebar = " ".join(texts(at.sidebar.markdown))
        assert f"**Model:** nvidia / {MODEL}" in sidebar
        labels = [b.label for b in at.sidebar.button]
        assert len(labels) == 6 and labels[-1] == "Clear chat"
        assert [label.split(":")[0] for label in labels[:5]] == [
            "Narrative", "Numeric", "Mixed", "Season list", "Not in the guide",
        ]
        assert fake.builds == 0  # the agent is only built when a question is asked

    def test_spinner_text_mentions_up_to_a_minute(self) -> None:
        assert "answers can take up to a minute" in APP.read_text()


class TestAsk:
    def test_question_shows_answer_steps_pages_and_timing(self, fake: FakeAgent) -> None:
        at = ask_in(run_app(), "Who won the 2026 Finals?")

        assert [m.name for m in at.chat_message] == ["user", "assistant"]
        assert "Answer to: Who won the 2026 Finals? (PDF p. 180)." in texts(at.markdown)
        assert "Answered in 7.8s" in texts(at.caption)
        assert at.expander[0].label == "How I got this"
        assert at.code[0].value == "SELECT winner_team FROM nba_finals WHERE year = 2026"
        assert at.code[0].language == "sql"
        body = texts(at.markdown)
        assert "**2. search_guide:** 2026 Finals MVP" in body
        assert "**Verified pages:** PDF p. 180" in body
        assert texts(at.warning) == ["Cited but not found in any tool output: PDF p. 999"]
        assert "Model calls: 3 · Latency: 7.8s · Tokens: 8300 in / 400 out" in texts(at.caption)
        assert fake.calls == [("Who won the 2026 Finals?", [])]
        assert not at.chat_input[0].disabled

    def test_history_is_passed_and_agent_is_built_once(self, fake: FakeAgent) -> None:
        at = ask_in(run_app(), "First?")
        ask_in(at, "Second?")
        assert fake.calls[1] == ("Second?", [("First?", "Answer to: First? (PDF p. 180).")])
        assert fake.builds == 1
        assert len(at.chat_message) == 4

    def test_example_button_asks_that_question(self, fake: FakeAgent) -> None:
        at = run_app()
        at.sidebar.button[3].click().run()
        assert fake.calls[0][0] == "Who was on the 1996-97 All-NBA first team?"


class TestErrors:
    @pytest.mark.parametrize(
        ("error", "kind", "shown"),
        [
            (agent.QuotaExceededError("The nvidia LLM quota or rate limit was hit (HTTP 429)."), "error",
             "The model's rate limit or quota was reached. The nvidia LLM quota or rate limit was hit (HTTP 429)."),
            (agent.LLMTimeoutError("The nvidia model did not answer in time. Try again, or ask a narrower question."),
             "warning", "The nvidia model did not answer in time. Try again, or ask a narrower question."),
            (agent.ConfigError("The nvidia API rejected the LLM API key (HTTP 401). Check NVIDIA_API_KEY."), "error",
             "Configuration problem: The nvidia API rejected the LLM API key (HTTP 401). Check NVIDIA_API_KEY."),
            (RuntimeError("[500] Internal Server Error"), "error", "Something went wrong (RuntimeError). Please try again."),
        ],
    )
    def test_errors_are_friendly_and_chat_stays_usable(
        self, fake: FakeAgent, error: Exception, kind: str, shown: str
    ) -> None:
        def fail(question: str) -> AgentResult:
            raise error

        fake.reply = fail
        at = ask_in(run_app(), "q")
        assert texts(getattr(at, kind)) == [shown]
        assert not at.exception
        assert not at.chat_input[0].disabled
        fake.reply = full_result
        ask_in(at, "next")
        assert "Answer to: next (PDF p. 180)." in texts(at.markdown)
        assert fake.calls[1][1] == []  # failed turns are not sent as history

    def test_too_many_steps_shows_steps_gathered(self, fake: FakeAgent) -> None:
        steps = [ToolStep("search_guide", {"query": "1997 all nba"}, "--- PDF p. 107 ---", [107])]

        def fail(question: str) -> AgentResult:
            raise agent.TooManyStepsError("The agent needed more than 21 steps and was stopped.", steps=steps)

        fake.reply = fail
        at = ask_in(run_app(), "q")
        assert texts(at.error) == ["The agent needed more than 21 steps and was stopped."]
        assert at.expander[0].label == "How I got this"
        assert "**1. search_guide:** 1997 all nba" in texts(at.markdown)

    def test_missing_key_is_shown_without_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_EMBED_API_KEY", "paid-secret-value")

        def no_key() -> object:
            raise agent.MissingConfigError("Missing required environment variable 'NVIDIA_API_KEY'.")

        monkeypatch.setattr(agent, "build_agent", no_key)
        at = ask_in(run_app(), "q")
        [message] = texts(at.error)
        assert message == "Configuration problem: Missing required environment variable 'NVIDIA_API_KEY'."
        page = " ".join(texts(at.markdown) + texts(at.caption) + texts(at.error))
        assert "secret" not in page


class TestSessionControls:
    def test_clear_chat_empties_history_but_keeps_the_count(self, fake: FakeAgent) -> None:
        at = ask_in(run_app(), "q1")
        at.sidebar.button[-1].click().run()
        assert len(at.chat_message) == 0
        assert at.session_state.messages == []
        assert at.session_state.questions_asked == 1
        ask_in(at, "q2")
        assert fake.calls[1] == ("q2", [])

    def test_question_cap(self, fake: FakeAgent, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_MAX_QUESTIONS", "2")
        at = ask_in(ask_in(run_app(), "q1"), "q2")
        assert at.chat_input[0].disabled
        assert all(b.disabled for b in at.sidebar.button[:5])
        assert "reached its limit of 2 questions" in texts(at.info)[0]
        assert "Questions this session: 2 / 2" in texts(at.sidebar.caption)
        with pytest.raises(AppTestError, match="disabled"):
            at.sidebar.button[0].click()
        with pytest.raises(AppTestError, match="disabled"):
            at.chat_input[0].set_value("q3")
        assert [c[0] for c in fake.calls] == ["q1", "q2"]
