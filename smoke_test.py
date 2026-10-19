"""Live smoke test: a few eval questions through the full agent (LLM_PROVIDER key + PAID Gemini embedding key).

    python smoke_test.py --check          # show the plan; no API calls
    python smoke_test.py                  # asks for confirmation, then runs n2, t3, m1

Each question makes at most 6 LLM calls (5 tool calls, then a forced text answer; recursion_limit 21 is
a backstop) plus one embedding call per search_guide.
A quota or config error stops the run immediately so no further LLM calls are spent.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from langchain_core.runnables import Runnable

from agent import (
    TOOL_CALL_LIMIT,
    AgentError,
    AgentResult,
    ConfigError,
    QuotaExceededError,
    ask,
    build_agent,
    llm_label,
)
from eval_retrieval import DEFAULT_QUESTIONS, EvalQuestion, load_questions

DEFAULT_IDS: tuple[str, ...] = ("n2", "t3", "m1")
EXTRA_QUESTIONS: tuple[EvalQuestion, ...] = (
    # Smoke-only: the single-year phrasing that used to hit the step cap (season rule: 1997 -> 1996-97).
    EvalQuestion("a4", "numeric", "Who were the all nba first team players in 1997", [107], verified=False),
)
MAX_LLM_CALLS_PER_QUESTION = TOOL_CALL_LIMIT + 1


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live smoke test of the agent on a few eval questions.")
    parser.add_argument("--ids", nargs="+", default=list(DEFAULT_IDS), help="Question ids from eval_questions.yaml")
    parser.add_argument("--delay", type=float, default=5.0, help="Seconds between questions (rate limits)")
    parser.add_argument("--check", action="store_true", help="Show the plan only; no API calls")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args(argv)

    by_id = {q.id: q for q in [*load_questions(DEFAULT_QUESTIONS), *EXTRA_QUESTIONS]}
    unknown = [i for i in args.ids if i not in by_id]
    if unknown:
        print(f"Unknown question ids: {unknown}", file=sys.stderr)
        return 2
    questions = [by_id[i] for i in args.ids]

    print(f"LLM: {llm_label()}")
    print(f"{len(questions)} questions: {', '.join(args.ids)}; up to "
          f"{len(questions) * MAX_LLM_CALLS_PER_QUESTION} LLM calls on the LLM key plus embedding calls on the PAID Gemini key.")
    if args.check:
        return 0
    if not args.yes and input("Run live now? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return 1

    try:
        graph = build_agent()
    except AgentError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 3
    return run(graph, questions, args.delay)


if __name__ == "__main__":
    sys.exit(main())
