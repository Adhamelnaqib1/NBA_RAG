"""Offline tests for smoke_test.py: the agent is scripted, never Gemini."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

import smoke_test
from agent import (
    AgentResult,
    ConfigError,
    QuotaExceededError,
    ToolStep,
    TooManyStepsError,
    build_agent,
)
from eval_retrieval import EvalQuestion
from test_agent import ScriptedChatModel, scripted, tool_call

Q = EvalQuestion("t1", "numeric", "Who was the 10th pick?", [52], verified=True)


class TestRun:
    def test_real_graph_with_fake_model(self, capsys: pytest.CaptureFixture[str]) -> None:
        model = scripted(
            tool_call("run_sql", {"sql": "SELECT 52 AS source_page"}, "c1"),
            AIMessage(content="Answer (PDF p. 52)."),
        )
        assert smoke_test.run(build_agent(llm=model), [Q], delay=0) == 0
        out = capsys.readouterr().out
        assert "tools: run_sql | tool calls: 1 | model calls: 2 | latency:" in out
        assert "cited: [52] | verified: [52] | unverified: []" in out
        assert "expected page (any of [52]): OK [52]" in out
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
        out = capsys.readouterr().out
        assert "LLM: nvidia / nvidia/nemotron-3.5-super-vl-120b-a12b" in out
        assert "3 questions: n2, t3, m1; up to 15 LLM calls on the LLM key" in out

    def test_unknown_id(self) -> None:
        assert smoke_test.main(["--check", "--ids", "zz"]) == 2

    def test_declined_confirmation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(smoke_test, "build_agent", MagicMock(side_effect=AssertionError("no agent")))
        monkeypatch.setattr("builtins.input", lambda prompt: "")
        assert smoke_test.main([]) == 1

    def test_missing_key_is_config_error(self) -> None:
        assert smoke_test.main(["--yes"]) == 3


def test_any_expected_page_is_enough_and_missing_is_flagged() -> None:
    q = EvalQuestion("a2", "numeric", "LeBron first team?", [111, 109])
    ok = smoke_test.format_outcome(q, AgentResult("13 (PDF p. 109).", pages=[109], verified_pages=[109]))
    missing = smoke_test.format_outcome(q, AgentResult("13 (PDF p. 108).", pages=[108], verified_pages=[108]))
    assert "expected page (any of [111, 109]): OK [109]" in ok
    assert "expected page (any of [111, 109]): MISSING" in missing


def test_extra_1997_question_is_selectable(capsys: pytest.CaptureFixture[str]) -> None:
    assert smoke_test.main(["--check", "--ids", "a1", "a2", "a3", "a4"]) == 0
    assert "4 questions: a1, a2, a3, a4; up to 20 LLM calls" in capsys.readouterr().out


def test_root_cause_reports_innermost_message() -> None:
    try:
        try:
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota generate_requests_per_day limit: 20")
        except RuntimeError as inner:
            raise QuotaExceededError("quota hit") from inner
    except QuotaExceededError as outer:
        assert smoke_test.root_cause(outer) == (
            "RuntimeError: 429 RESOURCE_EXHAUSTED quota generate_requests_per_day limit: 20"
        )


# ---------------------------------------------------------------------------------------------
# Model comparison (--models), offline: scripted fake chat models through the real agent graph
# ---------------------------------------------------------------------------------------------

Q1 = EvalQuestion("t1", "numeric", "Who was the 10th pick?", [52], verified=True)
Q2 = EvalQuestion("n2", "narrative", "Who won 2025-26?", [6, 180], verified=True)


def _scripted_build(per_model: dict[str, Callable[[], Any]], seen_env: list[dict[str, str | None]]) -> Callable[[], Any]:
    """Fake build_agent: picks the scripted model by the LLM_MODEL env var, records the overrides it saw."""

    def build() -> Any:
        seen_env.append({k: os.environ.get(k) for k in ("LLM_PROVIDER", "LLM_MODEL", "LLM_MAX_TOKENS")})
        return build_agent(llm=per_model[os.environ["LLM_MODEL"]]())

    return build


class TestCompareModels:
    @pytest.fixture(autouse=True)
    def no_sleep(self, monkeypatch: pytest.MonkeyPatch) -> list[float]:
        sleeps: list[float] = []
        monkeypatch.setattr(smoke_test.time, "sleep", sleeps.append)
        return sleeps

    def test_runs_each_model_with_env_overrides_and_restores_env(
        self, monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
    ) -> None:
        monkeypatch.setenv("LLM_MODEL", "original-model")
        seen: list[dict[str, str | None]] = []
        per_model = {
            "model-a": lambda: scripted(
                tool_call("run_sql", {"sql": "SELECT 52 AS source_page"}, "a1"), AIMessage(content="A1 (PDF p. 52)."),
                tool_call("run_sql", {"sql": "SELECT 180 AS source_page"}, "a2"), AIMessage(content="A2 (PDF p. 180)."),
            ),
            "model-b": lambda: scripted(AIMessage(content="B1 no citation."), AIMessage(content="B2 (PDF p. 999).")),
        }
        monkeypatch.setattr(smoke_test, "build_agent", _scripted_build(per_model, seen))

        outcomes, stopped = smoke_test.compare_models(["model-a", "model-b"], [Q1, Q2], sleep=3, max_tokens=8192)

        assert stopped == ""
        assert seen == [
            {"LLM_PROVIDER": "nvidia", "LLM_MODEL": "model-a", "LLM_MAX_TOKENS": "8192"},
            {"LLM_PROVIDER": "nvidia", "LLM_MODEL": "model-b", "LLM_MAX_TOKENS": "8192"},
        ]
        assert os.environ["LLM_MODEL"] == "original-model" and "LLM_MAX_TOKENS" not in os.environ
        assert no_sleep == [3, 3, 3]  # between the 4 questions, not before the first
        assert [(o.model, o.question.id, o.expected_ok) for o in outcomes] == [
            ("model-a", "t1", True), ("model-a", "n2", True), ("model-b", "t1", False), ("model-b", "n2", False),
        ]
        assert outcomes[3].result is not None and outcomes[3].result.unverified_pages == [999]

    def test_table_and_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        per_model = {
            "model-a": lambda: scripted(
                tool_call("run_sql", {"sql": "SELECT 52 AS source_page"}, "a1"),
                AIMessage(content="A1 (PDF p. 52).", usage_metadata={"input_tokens": 100, "output_tokens": 10,
                                                                     "total_tokens": 110}),
            ),
        }
        monkeypatch.setattr(smoke_test, "build_agent", _scripted_build(per_model, []))
        outcomes, _ = smoke_test.compare_models(["model-a"], [Q1], sleep=0)

        table = smoke_test.format_comparison(outcomes)
        row = table.splitlines()[2]
        assert row.startswith("| t1 | model-a | OK | run_sql | 1 | 2 | ")
        assert row.endswith(" | 100/10 | [52] | [] | - |")
        summary = smoke_test.format_summary(outcomes, ["model-a"])
        assert "| model-a | 1/1 | 1/1 | 2.0 | " in summary and "| 100/10 | 0 | 0 |" in summary
        answers = smoke_test.format_answers(outcomes)
        assert "- **model-a**: A1 (PDF p. 52)." in answers and "    - run_sql: SELECT 52 AS source_page" in answers

    def test_model_that_keeps_requesting_tools_is_counted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ScriptedChatModel ignores tool_choice='none', so it runs into the blocked calls and the backstop."""

        def endless() -> Any:
            def forever() -> Iterator[AIMessage]:
                n = 0
                while True:
                    n += 1
                    yield tool_call("run_sql", {"sql": "SELECT 1"}, f"c{n}")

            return ScriptedChatModel(messages=forever())

        monkeypatch.setattr(smoke_test, "build_agent", _scripted_build({"stubborn": endless}, []))
        outcomes, stopped = smoke_test.compare_models(["stubborn"], [Q1], sleep=0)

        assert stopped == ""
        [o] = outcomes
        assert o.backstop and o.error == "recursion backstop hit"
        assert o.blocked_calls == 3 and len(o.steps) == 6  # 3 SQL ran, then SQL-budget + 2 total refusals
        summary = smoke_test.format_summary(outcomes, ["stubborn"])
        assert summary.splitlines()[2] == "| stubborn | 0/1 | 0/1 | - | - | 0/0 | 1 | 1 |"

    def test_quota_error_stops_everything(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        def fake_ask(graph: Any, question: str) -> AgentResult:
            calls.append(os.environ["LLM_MODEL"])
            try:
                raise RuntimeError("[429] Too Many Requests: credits exhausted")
            except RuntimeError as inner:
                raise QuotaExceededError("The nvidia LLM quota or rate limit was hit (HTTP 429).") from inner

        monkeypatch.setattr(smoke_test, "build_agent", lambda: MagicMock())
        monkeypatch.setattr(smoke_test, "ask", fake_ask)
        outcomes, stopped = smoke_test.compare_models(["model-a", "model-b"], [Q1, Q2], sleep=0)

        assert calls == ["model-a"]
        assert len(outcomes) == 1 and outcomes[0].error.startswith("quota:")
        assert "model-a on t1" in stopped and "RuntimeError: [429] Too Many Requests: credits exhausted" in stopped

    def test_config_error_skips_that_model_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_ask(graph: Any, question: str) -> AgentResult:
            if os.environ["LLM_MODEL"] == "bad-model":
                raise ConfigError("Model 'bad-model' was not found by the nvidia API (HTTP 404). Check LLM_MODEL.")
            return AgentResult("ok (PDF p. 52).", pages=[52], verified_pages=[52])

        monkeypatch.setattr(smoke_test, "build_agent", lambda: MagicMock())
        monkeypatch.setattr(smoke_test, "ask", fake_ask)
        outcomes, stopped = smoke_test.compare_models(["bad-model", "good-model"], [Q1, Q2], sleep=0)

        assert stopped == ""
        assert [o.error for o in outcomes] == [
            "config: Model 'bad-model' was not found by the nvidia API (HTTP 404). Check LLM_MODEL.",
            "skipped (config error)", "", "",
        ]
        assert [o.expected_ok for o in outcomes] == [False, False, True, False]

    def test_build_config_error_and_unexpected_errors_are_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def build() -> Any:
            if os.environ["LLM_MODEL"] == "no-key-model":
                raise ConfigError("Missing required environment variable 'NVIDIA_API_KEY'.")
            return MagicMock()

        def fake_ask(graph: Any, question: str) -> AgentResult:
            raise RuntimeError("[400] Bad Request\nunsupported parameter: tool_choice")

        monkeypatch.setattr(smoke_test, "build_agent", build)
        monkeypatch.setattr(smoke_test, "ask", fake_ask)
        outcomes, stopped = smoke_test.compare_models(["no-key-model", "glm"], [Q1], sleep=0)

        assert stopped == ""
        assert outcomes[0].error == "config: Missing required environment variable 'NVIDIA_API_KEY'."
        assert outcomes[1].error == "RuntimeError: [400] Bad Request unsupported parameter: tool_choice"
        assert "| config: Missing required" in smoke_test.format_comparison(outcomes)


class TestCompareCli:
    def test_check_plan_for_models(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert smoke_test.main(["--check", "--models", "m-a, m-b", "--max-tokens", "8192",
                                "--ids", "n2", "t4", "m1", "a1", "a2", "a4"]) == 0
        out = capsys.readouterr().out
        assert "LLM: nvidia / m-a, m-b" in out
        assert "max tokens override: 8192" in out
        assert "6 questions: n2, t4, m1, a1, a2, a4; up to 60 LLM calls" in out

    @pytest.mark.parametrize("argv", [["--models", " , "], ["--max-tokens", "0"]])
    def test_bad_arguments(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit):
            smoke_test.main(["--check", *argv])

    def test_live_comparison_prints_table_summary_answers(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(smoke_test.time, "sleep", lambda s: None)
        monkeypatch.setattr(smoke_test, "build_agent", lambda: MagicMock())
        monkeypatch.setattr(smoke_test, "ask", lambda g, q: AgentResult("Answer (PDF p. 52).", pages=[52],
                                                                         verified_pages=[52], model_calls=2))
        assert smoke_test.main(["--yes", "--models", "m-a,m-b", "--ids", "t1"]) == 0
        out = capsys.readouterr().out
        assert "## Comparison" in out and "## Summary" in out and "## Answers" in out
        assert "| m-a | 1/1 | 1/1 | 2.0 |" in out and "- **m-b**: Answer (PDF p. 52)." in out

    def test_single_model_max_tokens_override_is_temporary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[str | None] = []

        def build() -> Any:
            seen.append(os.environ.get("LLM_MAX_TOKENS"))
            return MagicMock()

        monkeypatch.setattr(smoke_test, "build_agent", build)
        monkeypatch.setattr(smoke_test, "run", lambda graph, questions, delay: 0)
        assert smoke_test.main(["--yes", "--max-tokens", "2048", "--ids", "t1"]) == 0
        assert seen == ["2048"] and "LLM_MAX_TOKENS" not in os.environ


def test_env_overrides_restores_and_removes() -> None:
    os.environ["SMOKE_TEST_VAR"] = "before"
    try:
        with smoke_test.env_overrides({"SMOKE_TEST_VAR": "during", "SMOKE_TEST_NEW": "x"}):
            assert os.environ["SMOKE_TEST_VAR"] == "during" and os.environ["SMOKE_TEST_NEW"] == "x"
        assert os.environ["SMOKE_TEST_VAR"] == "before" and "SMOKE_TEST_NEW" not in os.environ
        with smoke_test.env_overrides({"SMOKE_TEST_VAR": None}):
            assert "SMOKE_TEST_VAR" not in os.environ
        assert os.environ["SMOKE_TEST_VAR"] == "before"
    finally:
        os.environ.pop("SMOKE_TEST_VAR", None)


def test_outcomes_are_reported_as_they_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted comparison keeps every finished result: each one is reported before the next starts."""
    reported: list[str] = []
    started: list[str] = []

    def fake_ask(graph: Any, question: str) -> AgentResult:
        started.append(f"{os.environ['LLM_MODEL']}:{question}")
        assert len(reported) == len(started) - 1  # the previous outcome was already reported
        return AgentResult("ok (PDF p. 52).", pages=[52], verified_pages=[52], model_calls=2)

    monkeypatch.setattr(smoke_test.time, "sleep", lambda s: None)
    monkeypatch.setattr(smoke_test, "build_agent", lambda: MagicMock())
    monkeypatch.setattr(smoke_test, "ask", fake_ask)
    smoke_test.compare_models(["m-a", "m-b"], [Q1, Q2], sleep=0,
                              on_outcome=lambda o: reported.append(f"{o.model}:{o.question.id}"))
    assert reported == ["m-a:t1", "m-a:n2", "m-b:t1", "m-b:n2"]


def test_format_progress_shows_fields_and_errors() -> None:
    ok = smoke_test.Outcome("m-a", Q1, result=AgentResult(
        "Brayden Burries (PDF p. 52).", steps=[ToolStep("run_sql", {"sql": "SELECT 1"}, "| 1 |")], pages=[52],
        verified_pages=[52], seconds=4.25, model_calls=2, input_tokens=5000, output_tokens=200,
    ))
    ok.steps = ok.result.steps if ok.result else []
    text = smoke_test.format_progress(ok)
    assert text.startswith("## t1 on m-a: expected page OK | tools: run_sql | tool calls: 1 | model calls: 2 "
                           "| latency: 4.2s | tokens: 5000/200")
    assert "cited: [52] | verified: [52] | unverified: []" in text and "answer: Brayden Burries (PDF p. 52)." in text
    failed = smoke_test.format_progress(smoke_test.Outcome("m-b", Q1, error="ReadTimeout: timed out"))
    assert failed == "## t1 on m-b: expected page MISSING | error: ReadTimeout: timed out"
