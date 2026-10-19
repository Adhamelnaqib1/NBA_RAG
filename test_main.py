"""Offline tests for main.py: the agent is a scripted fake model, never Gemini."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

import main
from agent import (
    AgentResult,
    QuotaExceededError,
    ToolStep,
    TooManyStepsError,
    build_agent,
)
from test_agent import scripted, tool_call


def _patch_build(monkeypatch: pytest.MonkeyPatch, factory: Callable[[], Runnable[Any, Any]]) -> None:
    monkeypatch.setattr(main, "build_agent", factory)


def _fixed_ask(monkeypatch: pytest.MonkeyPatch, outcome: AgentResult | Exception) -> MagicMock:
    mock = MagicMock(side_effect=outcome) if isinstance(outcome, Exception) else MagicMock(return_value=outcome)
    monkeypatch.setattr(main, "ask", mock)
    monkeypatch.setattr(main, "build_agent", lambda: MagicMock())
    return mock


class TestOneShot:
    def test_answers_with_real_graph_and_fake_model(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        model = scripted(
            tool_call("run_sql", {"sql": "SELECT 1 AS one"}, "c1"),
            AIMessage(content="One (PDF p. 52)."),
        )
        _patch_build(monkeypatch, lambda: build_agent(llm=model))

        assert main.main(["-v", "What", "is", "one?"]) == main.EXIT_OK
        out = capsys.readouterr().out
        assert out.startswith("One (PDF p. 52).")
        assert "[1] run_sql: SELECT 1 AS one" in out
        assert "cited: PDF p. 52" in out
        # SELECT 1 returns no source_page, so the citation is flagged.
        assert "WARNING: cited but not in any tool output: PDF p. 52" in out

    def test_not_verbose_prints_only_answer(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = AgentResult("Answer.", [ToolStep("run_sql", {"sql": "SELECT 1"}, "| 1 |")], [], 0.5)
        _fixed_ask(monkeypatch, result)
        assert main.main(["q"]) == main.EXIT_OK
        assert capsys.readouterr().out == "Answer.\n"

    @pytest.mark.parametrize(
        ("error", "code", "text"),
        [
            (QuotaExceededError("quota hit"), main.EXIT_QUOTA, "Quota: quota hit"),
            (TooManyStepsError("too many"), main.EXIT_TOO_MANY_STEPS, "Stopped: too many"),
            (ValueError("Question is empty."), main.EXIT_ERROR, "Error: Question is empty."),
        ],
    )
    def test_errors_give_friendly_message_and_exit_code(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
        error: Exception, code: int, text: str,
    ) -> None:
        _fixed_ask(monkeypatch, error)
        assert main.main(["q"]) == code
        captured = capsys.readouterr()
        assert text in captured.err
        assert "Traceback" not in captured.err

    def test_too_many_steps_prints_steps_so_far(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        steps = [ToolStep("search_guide", {"query": "1997 all-nba"}, "--- PDF p. 104 ---")]
        _fixed_ask(monkeypatch, TooManyStepsError("too many", steps=steps))
        assert main.main(["q"]) == main.EXIT_TOO_MANY_STEPS
        err = capsys.readouterr().err
        assert "Tool calls made before stopping:" in err
        assert "[1] search_guide: 1997 all-nba" in err

    def test_missing_key_exits_with_config_code(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main.main(["q"]) == main.EXIT_CONFIG
        assert "NVIDIA_API_KEY" in capsys.readouterr().err


class TestInteractive:
    def test_loop_keeps_history_and_survives_errors(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        outcomes: list[AgentResult | Exception] = [
            AgentResult("A1"), QuotaExceededError("slow down"), AgentResult("A3"),
        ]
        mock = _fixed_ask(monkeypatch, outcomes[0])
        mock.side_effect = outcomes
        inputs = iter(["q1", "", "q2", "q3", "exit"])
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        assert main.main([]) == main.EXIT_OK

        histories = [call.kwargs["history"] for call in mock.call_args_list]
        assert [c.args[1] for c in mock.call_args_list] == ["q1", "q2", "q3"]
        assert histories[2] == [("q1", "A1")]
        captured = capsys.readouterr()
        assert "A1" in captured.out and "A3" in captured.out
        assert "Quota: slow down" in captured.err

    def test_eof_exits_cleanly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fixed_ask(monkeypatch, AgentResult("x"))

        def eof(prompt: str) -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", eof)
        assert main.main([]) == main.EXIT_OK
