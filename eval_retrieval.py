"""Retrieval-only eval: hit@k and MRR of tools.retrieve over eval_questions.yaml. No LLM calls.

Each question makes one query embedding with the PAID embedding key. A question that misses is
re-run once before it counts as a miss, and both runs are reported.

    python eval_retrieval.py --check       # validate the questions file and show the plan; no API calls
    python eval_retrieval.py               # live run (asks for confirmation)
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CATEGORIES: tuple[str, ...] = ("narrative", "numeric", "mixed")
DEFAULT_QUESTIONS = Path("eval_questions.yaml")

SearchFn = Callable[[str, int], list[int]]
"""(query, k) -> PDF pages of the top-k retrieved chunks, closest first."""


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    category: str
    question: str
    expected_pages: list[int]
    verified: bool = False


@dataclass(frozen=True)
class RunResult:
    """One retrieval run: pages of the top-k chunks and the 1-based rank of the first expected page."""

    pages: list[int]
    rank: int | None

    @property
    def hit(self) -> bool:
        return self.rank is not None

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.rank if self.rank else 0.0


@dataclass
class QuestionResult:
    question: EvalQuestion
    runs: list[RunResult] = field(default_factory=list)

    @property
    def first(self) -> RunResult:
        return self.runs[0]

    @property
    def final(self) -> RunResult:
        """The run that decides the outcome: the re-run if there was one."""
        return self.runs[-1]


def load_questions(path: str | Path = DEFAULT_QUESTIONS) -> list[EvalQuestion]:
    """Loads and validates the questions file.

    Raises:
        ValueError: On a missing field, unknown category, duplicate id, or bad page list.
    """
    with open(path, encoding="utf-8") as f:
        raw: list[dict[str, Any]] = (yaml.safe_load(f) or {}).get("questions") or []

    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    for i, item in enumerate(raw, start=1):
        qid = str(item.get("id", "")).strip()
        where = f"question #{i} ({qid or 'no id'})"
        if not qid or qid in seen:
            raise ValueError(f"{where}: missing or duplicate id")
        if item.get("category") not in CATEGORIES:
            raise ValueError(f"{where}: category must be one of {CATEGORIES}")
        text = str(item.get("question", "")).strip()
        if not text:
            raise ValueError(f"{where}: empty question")
        pages = item.get("expected_pages")
        if not isinstance(pages, list) or not all(isinstance(p, int) and p > 0 for p in pages):
            raise ValueError(f"{where}: expected_pages must be a list of positive PDF page numbers")
        seen.add(qid)
        questions.append(EvalQuestion(qid, item["category"], text, pages, bool(item.get("verified", False))))
    return questions


def score_run(pages: list[int], expected: list[int]) -> RunResult:
    rank = next((i for i, p in enumerate(pages, start=1) if p in expected), None)
    return RunResult(pages=pages, rank=rank)


def evaluate(
    questions: list[EvalQuestion],
    search: SearchFn,
    k: int = 5,
    rerun_misses: bool = True,
) -> list[QuestionResult]:
    """Scores every question that has expected pages; a miss is re-run once when `rerun_misses`."""
    results: list[QuestionResult] = []
    for q in questions:
        if not q.expected_pages:
            logger.warning("Skipping %s: no expected_pages", q.id)
            continue
        result = QuestionResult(q, [score_run(search(q.question, k), q.expected_pages)])
        if rerun_misses and not result.first.hit:
            result.runs.append(score_run(search(q.question, k), q.expected_pages))
        results.append(result)
    return results


@dataclass(frozen=True)
class Summary:
    n: int
    hit_first: float
    mrr_first: float
    hit_final: float
    mrr_final: float


def summarize(results: list[QuestionResult]) -> Summary:
    n = len(results)
    if n == 0:
        return Summary(0, 0.0, 0.0, 0.0, 0.0)
    return Summary(
        n=n,
        hit_first=sum(r.first.hit for r in results) / n,
        mrr_first=sum(r.first.reciprocal_rank for r in results) / n,
        hit_final=sum(r.final.hit for r in results) / n,
        mrr_final=sum(r.final.reciprocal_rank for r in results) / n,
    )


def _run_cell(run: RunResult) -> str:
    return f"{'hit@' + str(run.rank) if run.hit else 'miss'} {run.pages}"


def report(results: list[QuestionResult], k: int) -> str:
    """Markdown report: one row per question with both runs, then summaries overall and per category."""
    lines = [
        f"| id | category | verified | expected | run 1 (top-{k} pages) | run 2 (re-run of misses) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        rerun = _run_cell(r.runs[1]) if len(r.runs) > 1 else "-"
        q = r.question
        lines.append(
            f"| {q.id} | {q.category} | {'yes' if q.verified else 'NO'} | {q.expected_pages} "
            f"| {_run_cell(r.first)} | {rerun} |"
        )

    lines += ["", f"| group | n | hit@{k} run 1 | MRR run 1 | hit@{k} final | MRR final |", "| --- | --- | --- | --- | --- | --- |"]
    groups: list[tuple[str, list[QuestionResult]]] = [("all", results)]
    groups += [(c, [r for r in results if r.question.category == c]) for c in CATEGORIES]
    for name, group in groups:
        s = summarize(group)
        lines.append(f"| {name} | {s.n} | {s.hit_first:.2f} | {s.mrr_first:.3f} | {s.hit_final:.2f} | {s.mrr_final:.3f} |")

    unverified = sum(not r.question.verified for r in results)
    if unverified:
        lines.append(f"\n{unverified} of {len(results)} questions have unverified expected_pages.")
    return "\n".join(lines)


def live_search(query: str, k: int) -> list[int]:
    """Real retrieval through tools.retrieve (PAID embedding key)."""
    from tools import retrieve

    return [hit.page for hit in retrieve(query, k=k)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval-only eval (hit@k, MRR). No LLM calls.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="Questions YAML")
    parser.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve (default: 5)")
    parser.add_argument("--no-rerun", action="store_true", help="Do not re-run misses")
    parser.add_argument("--only-verified", action="store_true", help="Score only questions with verified: true")
    parser.add_argument("--check", action="store_true", help="Validate the questions and print the plan; no API calls")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt before live calls")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    try:
        questions = load_questions(args.questions)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if args.only_verified:
        questions = [q for q in questions if q.verified]

    counts = ", ".join(f"{c}={sum(q.category == c for q in questions)}" for c in CATEGORIES)
    max_calls = len(questions) * (1 if args.no_rerun else 2)
    print(f"{len(questions)} questions ({counts}); "
          f"{len(questions)}-{max_calls} embedding calls on the PAID key, 0 LLM calls.")
    if args.check:
        return 0
    if not args.yes and input("Run live retrieval now? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return 1

    results = evaluate(questions, live_search, k=args.k, rerun_misses=not args.no_rerun)
    print(report(results, args.k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
