"""Offline tests for the NVIDIA NIM path: real ChatNVIDIA + real agent graph, fake HTTP session.

FakeSession replaces the requests.Session that ChatNVIDIA's client posts with. It returns real
requests.Response objects, so the library's own parsing and error formatting ("[429] Too Many Requests")
run exactly as they would against the live API, and every request payload is recorded.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import requests
from google.genai.errors import APIError
from langchain.agents.middleware import model_retry
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableBinding
from langchain_google_genai.chat_models import GoogleRateLimitError
from langchain_nvidia_ai_endpoints import ChatNVIDIA

import agent
import config
import tools
from agent import (
    ConfigError,
    QuotaExceededError,
    answer_text,
    ask,
    build_agent,
    strip_reasoning,
)
from tools import Hit

MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
FAKE_KEY = "nvapi-offline-test-key"


class NimError(Exception):
    """Stands in for the plain Exception ChatNVIDIA raises on HTTP errors."""


def http_response(status: int, body: dict[str, Any]) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(body).encode()
    response.headers["Content-Type"] = "application/json"
    response.url = "https://integrate.api.nvidia.com/v1/chat/completions"
    return response


def completion(
    content: str | None = None,
    tool_call: tuple[str, dict[str, Any], str] | None = None,
    reasoning: str | None = None,
    usage: tuple[int, int] = (100, 20),
) -> tuple[int, dict[str, Any]]:
    """An OpenAI-style chat.completion body as NIM returns it."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_call:
        name, args, call_id = tool_call
        message["tool_calls"] = [{"id": call_id, "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(args)}}]
    if reasoning:
        message["reasoning_content"] = reasoning
    return 200, {
        "id": "chatcmpl-test", "object": "chat.completion", "model": MODEL,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_call else "stop"}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)},
    }


def error(status: int, title: str, detail: str = "") -> tuple[int, dict[str, Any]]:
    return status, {"status": status, "title": title, "detail": detail or title}


class FakeSession:
    """Stands in for requests.Session: replays scripted (status, body) replies and records payloads."""

    def __init__(self, replies: Iterator[tuple[int, dict[str, Any]]] | list[tuple[int, dict[str, Any]]]) -> None:
        self.replies = iter(replies)
        self.payloads: list[dict[str, Any]] = []
        self.auth_headers: list[str] = []

    def post(self, url: str, headers: dict[str, str], json: dict[str, Any], timeout: float, **_: Any) -> requests.Response:
        self.payloads.append(json)
        self.auth_headers.append(headers.get("Authorization", ""))
        status, body = next(self.replies)
        return http_response(status, body)


def nvidia_llm(session: FakeSession) -> ChatNVIDIA:
    llm = ChatNVIDIA(model=MODEL, api_key=FAKE_KEY, max_completion_tokens=4096, timeout=120.0,
                     usage_telemetry_enabled=False)
    llm._client.get_session_fn = lambda: session
    return llm


@pytest.fixture(autouse=True)
def offline_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "missing.db")  # run_sql is not used here
    monkeypatch.setattr(model_retry.time, "sleep", lambda seconds: None)  # no real backoff waits
    monkeypatch.setattr(tools, "retrieve", lambda q, k: [Hit(52, 0.1, f"passage for {q}")])


def search_calls(n: int) -> list[tuple[int, dict[str, Any]]]:
    return [completion(tool_call=("search_guide", {"query": f"q{i}"}, f"call_{i}")) for i in range(n)]


class TestNvidiaAgentEndToEnd:
    def test_sixth_request_has_tool_choice_none_and_answer_is_clean(self) -> None:
        final = completion(content="<think>scratch work</think>AJ Dybantsa went first (PDF p. 52).",
                           reasoning="hidden chain of thought", usage=(300, 50))
        session = FakeSession([*search_calls(5), final])

        result = ask(build_agent(llm=nvidia_llm(session)), "Who was the first pick?")

        assert len(session.payloads) == 6
        assert [p.get("tool_choice") for p in session.payloads[:5]] == [None] * 5
        assert session.payloads[5]["tool_choice"] == "none"
        assert [t["function"]["name"] for t in session.payloads[5]["tools"]] == ["search_guide", "run_sql"]
        assert session.payloads[5]["model"] == MODEL
        assert session.payloads[5]["max_tokens"] == 4096
        assert result.answer == "AJ Dybantsa went first (PDF p. 52)."
        assert "scratch" not in result.answer and "hidden" not in result.answer
        assert result.model_calls == 6
        assert (result.input_tokens, result.output_tokens) == (5 * 100 + 300, 5 * 20 + 50)
        assert result.verified_pages == [52]

    def test_short_run_is_unaffected(self) -> None:
        session = FakeSession([*search_calls(1), completion(content="Answer (PDF p. 52).")])
        result = ask(build_agent(llm=nvidia_llm(session)), "q")
        assert [p.get("tool_choice") for p in session.payloads] == [None, None]
        assert result.answer == "Answer (PDF p. 52)." and result.model_calls == 2

    def test_key_only_in_authorization_header(self) -> None:
        session = FakeSession([completion(content="ok")])
        ask(build_agent(llm=nvidia_llm(session)), "q")
        assert session.auth_headers == [f"Bearer {FAKE_KEY}"]
        assert FAKE_KEY not in json.dumps(session.payloads)


class TestNvidiaErrors:
    def test_429_is_retried_then_succeeds(self) -> None:
        session = FakeSession([error(429, "Too Many Requests"), completion(content="ok")])
        result = ask(build_agent(llm=nvidia_llm(session)), "q")
        assert result.answer == "ok"
        assert len(session.payloads) == 2

    def test_persistent_429_becomes_quota_error_after_retries(self) -> None:
        session = FakeSession([error(429, "Too Many Requests")] * 10)
        with pytest.raises(QuotaExceededError, match=r"nvidia LLM quota or rate limit was hit \(HTTP 429\)"):
            ask(build_agent(llm=nvidia_llm(session), model_retries=2), "q")
        assert len(session.payloads) == 3  # first try + 2 retries

    def test_default_retry_count_for_nvidia_is_five(self) -> None:
        session = FakeSession([error(503, "Service Unavailable")] * 10)
        with pytest.raises(Exception, match=r"\[503\]"):
            ask(build_agent(llm=nvidia_llm(session)), "q")
        assert len(session.payloads) == 1 + agent.MODEL_RETRIES == 6

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_failure_is_config_error_naming_the_variable_only(self, status: int) -> None:
        session = FakeSession([error(status, "Unauthorized", f"bad key {FAKE_KEY}")] * 10)
        with pytest.raises(ConfigError) as exc_info:
            ask(build_agent(llm=nvidia_llm(session)), "q")
        message = str(exc_info.value)
        assert f"HTTP {status}" in message and "Check NVIDIA_API_KEY" in message
        assert FAKE_KEY not in message
        assert len(session.payloads) == 1  # never retried

    def test_unknown_model_is_config_error_naming_the_model(self) -> None:
        session = FakeSession([error(404, "Not Found", "Function not found for account")] * 10)
        with pytest.raises(ConfigError, match=f"Model '{MODEL}' was not found by the nvidia API"):
            ask(build_agent(llm=nvidia_llm(session)), "q")
        assert len(session.payloads) == 1

    def test_bad_request_is_not_retried_or_remapped(self) -> None:
        session = FakeSession([error(400, "Bad Request", "invalid tool_choice")] * 10)
        with pytest.raises(Exception, match=r"\[400\] Bad Request"):
            ask(build_agent(llm=nvidia_llm(session)), "q")
        assert len(session.payloads) == 1

    def test_timeout_is_retried(self) -> None:
        class TimesOutOnce(FakeSession):
            def post(self, *args: Any, **kwargs: Any) -> requests.Response:
                if not self.payloads:
                    self.payloads.append(kwargs["json"])
                    raise requests.Timeout("read timed out")
                return super().post(*args, **kwargs)

        session = TimesOutOnce([completion(content="ok")])
        assert ask(build_agent(llm=nvidia_llm(session)), "q").answer == "ok"
        assert len(session.payloads) == 2


class TestErrorMapping:
    """map_llm_error / llm_http_status / is_retryable_llm_error on simulated exceptions."""

    @pytest.mark.parametrize(
        ("exc", "status"),
        [
            (Exception("[429] Too Many Requests\nrate limited"), 429),
            (Exception("[401] Unauthorized\nPlease check or regenerate your API key."), 401),
            (Exception("  [404] Not Found\n"), 404),
            (APIError(429, {"error": {"message": "quota"}}, None), 429),
            (GoogleRateLimitError("429 RESOURCE_EXHAUSTED"), 429),
            (RuntimeError("no status here"), None),
        ],
    )
    def test_llm_http_status(self, exc: Exception, status: int | None) -> None:
        assert agent.llm_http_status(exc) == status

    def test_status_found_through_exception_chain(self) -> None:
        try:
            try:
                raise NimError("[429] Too Many Requests")
            except NimError as inner:
                raise RuntimeError("model call failed") from inner
        except RuntimeError as outer:
            assert agent.llm_http_status(outer) == 429
            assert isinstance(agent.map_llm_error(outer), QuotaExceededError)

    @pytest.mark.parametrize(
        "exc",
        [
            Exception("[429] Too Many Requests"),
            Exception("[402] Payment Required\ncredits exhausted"),
            Exception("upstream said Too Many Requests"),
            GoogleRateLimitError("429 RESOURCE_EXHAUSTED: quota exceeded"),
            APIError(429, {"error": {"message": "Quota", "status": "RESOURCE_EXHAUSTED"}}, None),
        ],
    )
    def test_quota_errors(self, exc: Exception) -> None:
        assert isinstance(agent.map_llm_error(exc), QuotaExceededError)

    @pytest.mark.parametrize(("provider", "var"), [("nvidia", "NVIDIA_API_KEY"), ("gemini", "GEMINI_LLM_API_KEY")])
    def test_auth_errors_name_variable_never_value(
        self, monkeypatch: pytest.MonkeyPatch, provider: str, var: str
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", provider)
        monkeypatch.setenv(var, "super-secret-value")
        mapped = agent.map_llm_error(Exception("[401] Unauthorized\nkey super-secret-value rejected"))
        assert isinstance(mapped, ConfigError)
        assert f"Check {var}" in str(mapped) and "super-secret-value" not in str(mapped)

    def test_gemini_typed_auth_and_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_MODEL", "gemini-9-ultra")
        auth = agent.map_llm_error(APIError(403, {"error": {"message": "denied"}}, None))
        missing = agent.map_llm_error(APIError(404, {"error": {"message": "models/gemini-9-ultra is not found"}}, None))
        assert isinstance(auth, ConfigError) and "GEMINI_LLM_API_KEY" in str(auth)
        assert isinstance(missing, ConfigError) and "Model 'gemini-9-ultra' was not found by the gemini API" in str(missing)

    def test_model_not_found_text_without_status(self) -> None:
        mapped = agent.map_llm_error(Exception("The model `nvidia/xyz` does not exist"))
        assert isinstance(mapped, ConfigError) and MODEL in str(mapped)

    @pytest.mark.parametrize("exc", [Exception("[500] Internal Server Error"), RuntimeError("boom"),
                                     Exception("[400] Bad Request")])
    def test_other_errors_are_not_mapped(self, exc: Exception) -> None:
        assert agent.map_llm_error(exc) is None

    @pytest.mark.parametrize(
        ("exc", "retry"),
        [
            (Exception("[429] Too Many Requests"), True), (Exception("[503] Service Unavailable"), True),
            (requests.Timeout("timed out"), True), (requests.ConnectionError("reset"), True),
            (Exception("[401] Unauthorized"), False), (Exception("[404] Not Found"), False),
            (Exception("[400] Bad Request"), False), (ValueError("bad args"), False),
        ],
    )
    def test_is_retryable(self, exc: Exception, retry: bool) -> None:
        assert agent.is_retryable_llm_error(exc) is retry


class TestReasoningStripping:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("<think>plan</think>Answer.", "Answer."),
            ("<THINK>\nplan\n</THINK>\n\nAnswer.", "Answer."),
            ("plan without opening tag</think>Answer.", "Answer."),
            ("Answer.<think>trailing unclosed", "Answer."),
            ("<think>a</think>Part 1 <think>b</think>part 2", "Part 1 part 2"),
            ("Plain answer (PDF p. 52).", "Plain answer (PDF p. 52)."),
            ("<think>only reasoning</think>", ""),
        ],
    )
    def test_strip_reasoning(self, text: str, expected: str) -> None:
        assert strip_reasoning(text) == expected

    def test_reasoning_content_blocks_are_ignored(self) -> None:
        msg = AIMessage(content=[
            {"type": "reasoning", "reasoning": "secret plan"},
            {"type": "thinking", "thinking": "more plan"},
            {"type": "text", "text": "<think>x</think>Final "},
            {"type": "text", "text": "answer."},
        ])
        assert answer_text(msg) == "Final answer."

    def test_additional_kwargs_reasoning_is_ignored(self) -> None:
        msg = AIMessage(content="Final answer.", additional_kwargs={"reasoning_content": "secret plan",
                                                                    "reasoning": "secret plan"})
        assert answer_text(msg) == "Final answer."

    def test_reasoning_only_answer_is_an_agent_error(self) -> None:
        session = FakeSession([completion(content="<think>I never answer</think>", reasoning="r")])
        with pytest.raises(agent.AgentError, match="without an answer"):
            ask(build_agent(llm=nvidia_llm(session)), "q")


class TestPayloadDirect:
    def test_bind_tools_none_payload(self) -> None:
        """The exact JSON ChatNVIDIA would send for a forced-answer request (no network)."""
        llm = nvidia_llm(FakeSession([]))
        forced = cast(RunnableBinding[Any, Any], llm.bind_tools([tools.search_guide, tools.run_sql], tool_choice="none"))
        _, payload, _ = llm._prepare_inputs_and_payload([AIMessage(content="hi")], None, stream=False, **forced.kwargs)
        assert payload["tool_choice"] == "none"
        assert {t["function"]["name"] for t in payload["tools"]} == {"search_guide", "run_sql"}

    def test_llm_label_has_provider_and_model_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert agent.llm_label() == f"nvidia / {MODEL}"
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        assert agent.llm_label() == "gemini / gemini-2.5-flash"
        monkeypatch.setenv("LLM_PROVIDER", "bogus")
        assert agent.llm_label() == "configured / unknown"
