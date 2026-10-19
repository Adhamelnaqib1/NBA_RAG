"""Live smoke test: a few eval questions through the full agent (LLM_PROVIDER key + PAID Gemini embedding key).

    python smoke_test.py --check          # show the plan; no API calls
    python smoke_test.py                  # asks for confirmation, then runs n2, t3, m1
    python smoke_test.py --models a,b --ids n2 t4   # same questions on each NVIDIA model, one comparison table

Each question makes at most TOTAL_TOOL_BUDGET + 1 = 5 LLM calls (4 tool calls, then a forced text answer;
the recursion limit is a backstop) plus one embedding call per search_guide.
A quota error stops the run immediately so no further LLM calls are spent; failures are never re-run.
LLM_MODEL / LLM_MAX_TOKENS overrides apply to this process only; .env is never edited.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import Runnable

from agent import (
    AgentError,
    AgentResult,
    ConfigError,
    QuotaExceededError,
    ToolStep,
    TooManyStepsError,
    ask,
    build_agent,
    llm_label,
    max_model_calls,
)
from eval_retrieval import DEFAULT_QUESTIONS, EvalQuestion, load_questions

DEFAULT_IDS: tuple[str, ...] = ("n2", "t3", "m1")
EXTRA_QUESTIONS: tuple[EvalQuestion, ...] = (
    # Smoke-only: the single-year phrasing that used to hit the step cap (season rule: 1997 -> 1996-97).
    EvalQuestion("a4", "numeric", "Who were the all nba first team players in 1997", [107], verified=False),
)
MAX_LLM_CALLS_PER_QUESTION = max_model_calls()


def format_outcome(q: EvalQuestion, result: AgentResult) -> str:
    tools_used = ", ".join(dict.fromkeys(s.name for s in result.steps)) or "none"
    # Any one expected page is enough: a table answer can cite either source page.
    expected_hit = sorted(set(result.verified_pages) & set(q.expected_pages))
    lines = [
        f"## {q.id} ({q.category}): {q.question}",
        (f"tools: {tools_used} | tool calls: {len(result.steps)} | model calls: {result.model_calls} "
         f"| latency: {result.seconds:.1f}s"),
        f"tokens: {result.input_tokens} in / {result.output_tokens} out",
        f"cited: {result.pages} | verified: {result.verified_pages} | unverified: {result.unverified_pages}",
        f"expected page (any of {q.expected_pages}): {'OK ' + str(expected_hit) if expected_hit else 'MISSING'}",
    ]
    lines += [f"  [{i}] {s.name}: {s.args.get('sql') or s.args.get('query') or s.args}"
              for i, s in enumerate(result.steps, start=1)]
    lines += ["answer:", result.answer]
    return "\n".join(lines)


def root_cause(exc: BaseException, limit: int = 1200) -> str:
    """The innermost chained exception's message (quota name and limit), shortened; carries no key values."""
    while exc.__cause__ is not None:
        exc = exc.__cause__
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text[:limit]}{'...' if len(text) > limit else ''}"


def run(graph: Runnable[Any, Any], questions: list[EvalQuestion], delay: float) -> int:
    """Runs each question, printing its outcome. Returns 0 if all answered, 1 otherwise."""
    failures = 0
    for i, q in enumerate(questions):
        if i and delay > 0:
            time.sleep(delay)
        try:
            print(format_outcome(q, ask(graph, q.question)) + "\n")
        except QuotaExceededError as e:
            print(f"## {q.id}: QUOTA - {e}\nProvider said: {root_cause(e)}\nStopping to avoid more LLM calls.")
            return 1
        except ConfigError as e:
            print(f"## {q.id}: CONFIG ERROR - {e}\nStopping: later questions would fail the same way.")
            return 1
        except AgentError as e:
            print(f"## {q.id}: FAILED ({type(e).__name__}) - {e}\n")
            failures += 1
    return 1 if failures else 0


# --------------------------------------------------------------------------------------------------
# Model comparison (--models)
# --------------------------------------------------------------------------------------------------

LIMIT_MESSAGE = "limit reached"


@contextmanager
def env_overrides(values: dict[str, str | None]) -> Iterator[None]:
    """Sets (or, for None, removes) environment variables in this process only, restoring them afterwards."""
    saved = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


def model_env(model: str, max_tokens: int | None) -> dict[str, str | None]:
    """Overrides for one comparison run: always the NVIDIA provider; LLM_MAX_TOKENS only when given."""
    env: dict[str, str | None] = {"LLM_PROVIDER": "nvidia", "LLM_MODEL": model}
    if max_tokens is not None:
        env["LLM_MAX_TOKENS"] = str(max_tokens)
    return env


@dataclass
class Outcome:
    """One (model, question) run in a comparison."""

    model: str
    question: EvalQuestion
    result: AgentResult | None = None
    error: str = ""
    backstop: bool = False
    steps: list[ToolStep] = field(default_factory=list)

    @property
    def expected_ok(self) -> bool:
        return self.result is not None and bool(set(self.result.verified_pages) & set(self.question.expected_pages))

    @property
    def blocked_calls(self) -> int:
        """Tool calls refused by the per-tool or total budget (ToolBudgetMiddleware)."""
        return sum(LIMIT_MESSAGE in s.output_preview for s in self.steps)


def compare_models(
    models: list[str],
    questions: list[EvalQuestion],
    sleep: float,
    max_tokens: int | None = None,
    on_outcome: Callable[[Outcome], None] | None = None,
) -> tuple[list[Outcome], str]:
    """Runs every question on every model, in order. Returns the outcomes and a stop reason ("" if none).

    `on_outcome` is called as soon as each (model, question) finishes, so an interrupted run keeps
    everything it completed.

    A quota error stops everything. A config error (rejected key, unknown model) skips that model's
    remaining questions. Any other failure is recorded and the run continues; nothing is re-run.
    """
    outcomes: list[Outcome] = []

    def finish(outcome: Outcome) -> None:
        outcomes.append(outcome)
        if on_outcome is not None:
            on_outcome(outcome)

    first = True
    for model in models:
        with env_overrides(model_env(model, max_tokens)):
            try:
                graph = build_agent()
            except ConfigError as e:
                for q in questions:
                    finish(Outcome(model, q, error=f"config: {e}"))
                continue
            skip_reason = ""
            for q in questions:
                if skip_reason:
                    finish(Outcome(model, q, error=f"skipped ({skip_reason})"))
                    continue
                if not first and sleep > 0:
                    time.sleep(sleep)
                first = False
                outcome = Outcome(model, q)
                try:
                    outcome.result = ask(graph, q.question)
                    outcome.steps = outcome.result.steps
                except QuotaExceededError as e:
                    outcome.error = f"quota: {e}"
                    finish(outcome)
                    return outcomes, f"{model} on {q.id}: {e} Provider said: {root_cause(e)}"
                except ConfigError as e:
                    outcome.error = f"config: {e}"
                    skip_reason = "config error"
                except TooManyStepsError as e:
                    outcome.error = "recursion backstop hit"
                    outcome.backstop = True
                    outcome.steps = e.steps
                except Exception as e:  # report boundary: record any provider/parameter failure, keep going
                    logging.getLogger(__name__).warning("%s on %s failed", model, q.id, exc_info=True)
                    outcome.error = root_cause(e, limit=300)
                finish(outcome)
    return outcomes, ""


def format_progress(o: Outcome) -> str:
    """One finished (model, question) run, printed immediately during a comparison."""
    r = o.result
    head = f"## {o.question.id} on {o.model}: expected page {'OK' if o.expected_ok else 'MISSING'}"
    if r is None:
        return f"{head} | error: {o.error}"
    tools_used = ", ".join(dict.fromkeys(s.name for s in o.steps)) or "none"
    lines = [
        (f"{head} | tools: {tools_used} | tool calls: {len(r.steps)} | model calls: {r.model_calls} "
         f"| latency: {r.seconds:.1f}s | tokens: {r.input_tokens}/{r.output_tokens}"),
        f"cited: {r.pages} | verified: {r.verified_pages} | unverified: {r.unverified_pages}"
        + (f" | blocked tool calls after limit: {o.blocked_calls}" if o.blocked_calls else ""),
    ]
    lines += [f"  [{i}] {s.name}: {s.args.get('sql') or s.args.get('query') or s.args}"
              for i, s in enumerate(o.steps, start=1)]
    lines += ["answer: " + r.answer, ""]
    return "\n".join(lines)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def format_comparison(outcomes: list[Outcome]) -> str:
    """One markdown row per (question, model)."""
    lines = [
        "| q | model | expected | tools | tool calls | model calls | latency | tokens in/out | verified | unverified | error |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for o in sorted(outcomes, key=lambda o: (o.question.id, o.model)):
        r = o.result
        tools_used = ", ".join(dict.fromkeys(s.name for s in o.steps)) or "-"
        if r is not None:
            cols = [f"{len(r.steps)}", f"{r.model_calls}", f"{r.seconds:.1f}s", f"{r.input_tokens}/{r.output_tokens}",
                    f"{r.verified_pages}", f"{r.unverified_pages}"]
        else:
            cols = [f"{len(o.steps)}" if o.steps else "-", "-", "-", "-", "-", "-"]
        expected = "OK" if o.expected_ok else "MISSING"
        lines.append(f"| {o.question.id} | {o.model} | {expected} | {tools_used} | " + " | ".join(cols)
                     + f" | {_cell(o.error) or '-'} |")
    return "\n".join(lines)


def format_summary(outcomes: list[Outcome], models: list[str]) -> str:
    """Per model: expected pages cited, averages over answered runs, token totals, limit/backstop counts."""
    lines = [
        "| model | expected cited | answered | avg model calls | avg latency | tokens in/out | backstop runs | runs with tool calls after limit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model in models:
        runs = [o for o in outcomes if o.model == model]
        answered = [o.result for o in runs if o.result is not None]
        n = len(answered)
        avg_calls = f"{sum(r.model_calls for r in answered) / n:.1f}" if n else "-"
        avg_latency = f"{sum(r.seconds for r in answered) / n:.1f}s" if n else "-"
        tokens = f"{sum(r.input_tokens for r in answered)}/{sum(r.output_tokens for r in answered)}"
        lines.append(
            f"| {model} | {sum(o.expected_ok for o in runs)}/{len(runs)} | {n}/{len(runs)} | {avg_calls} | "
            f"{avg_latency} | {tokens} | {sum(o.backstop for o in runs)} | {sum(o.blocked_calls > 0 for o in runs)} |"
        )
    return "\n".join(lines)


def format_answers(outcomes: list[Outcome]) -> str:
    """Final answers grouped by question, so each model's answer can be checked against SQL."""
    lines: list[str] = []
    for qid in dict.fromkeys(o.question.id for o in outcomes):
        group = [o for o in outcomes if o.question.id == qid]
        lines.append(f"### {qid}: {group[0].question.question}")
        for o in group:
            text = o.result.answer if o.result is not None else f"(no answer: {o.error})"
            lines.append(f"- **{o.model}**: {text}")
            lines += [f"    - {s.name}: {s.args.get('sql') or s.args.get('query') or s.args}" for s in o.steps]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live smoke test of the agent on a few eval questions.")
    parser.add_argument("--ids", nargs="+", default=list(DEFAULT_IDS), help="Question ids from eval_questions.yaml")
    parser.add_argument("--models", type=lambda v: [m.strip() for m in v.split(",") if m.strip()], default=None,
                        help="Comma-separated NVIDIA model ids to compare on the same questions")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override LLM_MAX_TOKENS for this run only")
    parser.add_argument("--sleep", type=float, default=3.0, help="Seconds between questions (default: 3)")
    parser.add_argument("--check", action="store_true", help="Show the plan only; no API calls")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args(argv)
    if args.max_tokens is not None and args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.models is not None and not args.models:
        parser.error("--models needs at least one model id")

    by_id = {q.id: q for q in [*load_questions(DEFAULT_QUESTIONS), *EXTRA_QUESTIONS]}
    unknown = [i for i in args.ids if i not in by_id]
    if unknown:
        print(f"Unknown question ids: {unknown}", file=sys.stderr)
        return 2
    questions = [by_id[i] for i in args.ids]

    models: list[str] | None = args.models
    runs = len(questions) * (len(models) if models else 1)
    print(f"LLM: nvidia / {', '.join(models)}" if models else f"LLM: {llm_label()}")
    if args.max_tokens is not None:
        print(f"max tokens override: {args.max_tokens}")
    print(f"{len(questions)} questions: {', '.join(args.ids)}; up to "
          f"{runs * MAX_LLM_CALLS_PER_QUESTION} LLM calls on the LLM key plus embedding calls on the PAID Gemini key.")
    if args.check:
        return 0
    if not args.yes and input("Run live now? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return 1

    if models:
        outcomes, stopped = compare_models(
            models, questions, args.sleep, args.max_tokens,
            on_outcome=lambda o: print(format_progress(o), flush=True),
        )
        print("\n## Comparison\n" + format_comparison(outcomes))
        print("\n## Summary\n" + format_summary(outcomes, models))
        print("\n## Answers\n" + format_answers(outcomes))
        if stopped:
            print(f"STOPPED on quota error: {stopped}")
            return 1
        return 0 if all(o.result is not None for o in outcomes) else 1

    with env_overrides({"LLM_MAX_TOKENS": str(args.max_tokens)} if args.max_tokens is not None else {}):
        try:
            graph = build_agent()
        except AgentError as e:
            print(f"Config error: {e}", file=sys.stderr)
            return 3
        return run(graph, questions, args.sleep)


if __name__ == "__main__":
    sys.exit(main())
