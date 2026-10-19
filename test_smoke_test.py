"""Offline tests for smoke_test.py: the agent is scripted, never Gemini."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

import smoke_test
from agent import (
    AgentResult,
    QuotaExceededError,
    ToolStep,
    TooManyStepsError,
    build_agent,
)
from eval_retrieval import EvalQuestion
from test_agent import scripted, tool_call

Q = EvalQuestion("t1", "numeric", "Who was the 10th pick?", [52], verified=True)


class TestRun:
    def test_real_graph_with_fake_model(self, capsys: pytest.CaptureFixture[str]) -> None:
        model = scripted(
            tool_call("run_sql", {"sql": "SELECT 52 AS source_page"}, "c1"),
            AIMessage(content="Answer (PDF p. 52)."),
        )
        assert smoke_test.run(build_agent(llm=model), [Q], delay=0) == 0
        out = capsys.readouterr().out
        assert "tools: run_sql | steps: 1 | latency:" in out
        assert "cited: [52] | verified: [52] | unverified: []" in out
        assert "expected pages [52] cited and verified: [52]" in out
        assert "  [1] run_sql: SELECT 52 AS source_page" in out

    def test_quota_stops_immediately(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        mock_ask = MagicMock(side_effect=QuotaExceededError("429"))
        monkeypatch.setattr(smoke_test, "ask", mock_ask)
        assert smoke_test.run(MagicMock(), [Q, Q, Q], delay=0) == 1
        assert mock_ask.call_count == 1
        assert "Stopping" in capsys.readouterr().out

    def test_other_failures_continue_and_return_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_ask = MagicMock(side_effect=[TooManyStepsError("x"), AgentResult("ok", [ToolStep("run_sql", {}, "")])])
        monkeypatch.setattr(smoke_test, "ask", mock_ask)
        assert smoke_test.run(MagicMock(), [Q, Q], delay=0) == 1
        assert mock_ask.call_count == 2


class TestMain:
    def test_check_prints_plan_without_building_agent(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(smoke_test, "build_agent", MagicMock(side_effect=AssertionError("no agent")))
        assert smoke_test.main(["--check"]) == 0
        assert "3 questions: n2, t3, m1; up to 24 LLM calls on the FREE key" in capsys.readouterr().out

    def test_unknown_id(self) -> None:
        assert smoke_test.main(["--check", "--ids", "zz"]) == 2

    def test_declined_confirmation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(smoke_test, "build_agent", MagicMock(side_effect=AssertionError("no agent")))
        monkeypatch.setattr("builtins.input", lambda prompt: "")
        assert smoke_test.main([]) == 1

    def test_missing_key_is_config_error(self) -> None:
        assert smoke_test.main(["--yes"]) == 3
