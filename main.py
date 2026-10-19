"""CLI for the NBA Guide agent.

    python main.py "Who was the first pick in the 2026 draft?"   # one-shot
    python main.py                                               # interactive (exit / quit / Ctrl-D)
    python main.py -v "..."                                      # also print tool steps
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from langchain_core.runnables import Runnable

import config
from agent import (
    AgentError,
    AgentResult,
    MissingConfigError,
    QuotaExceededError,
    TooManyStepsError,
    ask,
    build_agent,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 3
EXIT_QUOTA = 4
EXIT_TOO_MANY_STEPS = 5

QUIT_WORDS = frozenset({"exit", "quit", ":q"})


def format_result(result: AgentResult, verbose: bool) -> str:
    """The answer, followed by tool steps and timing when verbose."""
    lines = [result.answer]
    if verbose:
        lines.append("")
        for i, step in enumerate(result.steps, start=1):
            arg = step.args.get("sql") or step.args.get("query") or step.args
            lines.append(f"[{i}] {step.name}: {arg}")
            lines.append("    " + step.output_preview.replace("\n", "\n    "))
        pages = ", ".join(f"PDF p. {p}" for p in result.pages) or "none"
        lines.append(f"({len(result.steps)} tool calls, {result.seconds:.1f}s, cited: {pages})")
        if result.unverified_pages:
            unverified = ", ".join(f"PDF p. {p}" for p in result.unverified_pages)
            lines.append(f"WARNING: cited but not in any tool output: {unverified}")
    return "\n".join(lines)


def answer_once(graph: Runnable[Any, Any], question: str, verbose: bool,
                history: list[tuple[str, str]] | None = None) -> tuple[int, AgentResult | None]:
    """Asks one question, printing the answer or a friendly error. Returns (exit code, result)."""
    try:
        result = ask(graph, question, history=history)
    except QuotaExceededError as e:
        print(f"Quota: {e}", file=sys.stderr)
        return EXIT_QUOTA, None
    except TooManyStepsError as e:
        print(f"Stopped: {e}", file=sys.stderr)
        return EXIT_TOO_MANY_STEPS, None
    except (AgentError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_ERROR, None
    print(format_result(result, verbose))
    return EXIT_OK, result


def interactive(graph: Runnable[Any, Any], verbose: bool) -> int:
    """Question loop with conversation history. Errors are reported and the loop continues."""
    print(f"NBA Guide Q&A ({config.LLM_MODEL}). Type 'exit' to quit.")
    history: list[tuple[str, str]] = []
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return EXIT_OK
        if not question:
            continue
        if question.lower() in QUIT_WORDS:
            return EXIT_OK
        _, result = answer_once(graph, question, verbose, list(history))
        if result is not None:
            history.append((question, result.answer))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask questions about the 2026-27 NBA Guide.")
    parser.add_argument("question", nargs="*", help="Question to ask; omit for interactive mode")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show tool calls, timing and cited pages")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if not config.DB_PATH.exists():
        print(f"Warning: {config.DB_PATH} not found; run_sql will fail. "
              "Load it with: python extract_tables.py load --pdf 2026-27-NBA-Guide.pdf", file=sys.stderr)

    try:
        graph = build_agent()
    except MissingConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return EXIT_CONFIG

    if args.question:
        code, _ = answer_once(graph, " ".join(args.question), args.verbose)
        return code
    return interactive(graph, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
