"""Offline tests for eval_retrieval.py: retrieval is a scripted fake, never Gemini."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import eval_retrieval
from eval_retrieval import (
    EvalQuestion,
    evaluate,
    load_questions,
    report,
    score_run,
    summarize,
)


def _q(qid: str, pages: list[int], category: str = "narrative") -> EvalQuestion:
    return EvalQuestion(qid, category, f"question {qid}", pages, verified=False)


def scripted_search(answers: dict[str, list[list[int]]]) -> tuple[eval_retrieval.SearchFn, list[str]]:
    """Each call for a question returns its next scripted page list; also records the calls."""
    queues: dict[str, Iterator[list[int]]] = {q: iter(runs) for q, runs in answers.items()}
    calls: list[str] = []

    def search(query: str, k: int) -> list[int]:
        calls.append(query)
        return next(queues[query])[:k]

    return search, calls


class TestScoring:
    @pytest.mark.parametrize(
        ("pages", "expected", "rank", "rr"),
        [
            ([52, 7, 8], [52], 1, 1.0),
            ([7, 8, 180, 179], [179, 180], 3, 1 / 3),
            ([7, 8], [52], None, 0.0),
            ([], [52], None, 0.0),
        ],
    )
    def test_score_run(self, pages: list[int], expected: list[int], rank: int | None, rr: float) -> None:
        run = score_run(pages, expected)
        assert run.rank == rank
        assert run.hit is (rank is not None)
        assert run.reciprocal_rank == pytest.approx(rr)

    def test_summary_math(self) -> None:
        search, _ = scripted_search({
            "question a": [[1, 2]],          # hit@1
            "question b": [[9, 2]],          # hit@2
            "question c": [[9], [9]],        # miss, miss
            "question d": [[9], [4, 9]],     # miss, then hit@1 on the re-run
        })
        qs = [_q("a", [1]), _q("b", [2]), _q("c", [3]), _q("d", [4])]
        s = summarize(evaluate(qs, search, k=5))
        assert s.n == 4
        assert s.hit_first == pytest.approx(0.5)
        assert s.mrr_first == pytest.approx((1 + 0.5) / 4)
        assert s.hit_final == pytest.approx(0.75)
        assert s.mrr_final == pytest.approx((1 + 0.5 + 0 + 1) / 4)

    def test_empty_summary(self) -> None:
        assert summarize([]).n == 0


class TestRerun:
    def test_miss_is_rerun_once_and_both_runs_kept(self) -> None:
        search, calls = scripted_search({"question a": [[7, 8], [52, 7]]})
        [result] = evaluate([_q("a", [52])], search)
        assert calls == ["question a", "question a"]
        assert [r.hit for r in result.runs] == [False, True]
        assert result.final.rank == 1

    def test_hit_is_not_rerun(self) -> None:
        search, calls = scripted_search({"question a": [[52]]})
        [result] = evaluate([_q("a", [52])], search)
        assert calls == ["question a"]
        assert len(result.runs) == 1

    def test_double_miss_stays_a_miss(self) -> None:
        search, calls = scripted_search({"question a": [[1], [2]]})
        [result] = evaluate([_q("a", [52])], search)
        assert len(calls) == 2
        assert not result.final.hit

    def test_no_rerun_option(self) -> None:
        search, calls = scripted_search({"question a": [[1]]})
        [result] = evaluate([_q("a", [52])], search, rerun_misses=False)
        assert calls == ["question a"]
        assert not result.final.hit

    def test_k_is_passed_to_search(self) -> None:
        seen: list[int] = []

        def search(query: str, k: int) -> list[int]:
            seen.append(k)
            return [52]

        evaluate([_q("a", [52])], search, k=3)
        assert seen == [3]

    def test_question_without_pages_is_skipped(self) -> None:
        search, calls = scripted_search({})
        assert evaluate([_q("a", [])], search) == []
        assert calls == []


class TestReport:
    def test_report_shows_both_runs_and_groups(self) -> None:
        search, _ = scripted_search({"question a": [[7], [52]], "question b": [[180]]})
        results = evaluate([_q("a", [52]), _q("b", [180], "numeric")], search)
        text = report(results, k=5)
        assert "| a | narrative | NO | [52] | miss [7] | hit@1 [52] |" in text
        assert "| b | numeric | NO | [180] | hit@1 [180] | - |" in text
        assert "| all | 2 | 0.50 | 0.500 | 1.00 | 1.000 |" in text
        assert "| mixed | 0 |" in text
        assert "2 of 2 questions have unverified expected_pages." in text


class TestLoadQuestions:
    def test_repo_questions_file(self) -> None:
        qs = load_questions(Path(__file__).parent / "eval_questions.yaml")
        assert len(qs) == 15
        assert {c: sum(q.category == c for q in qs) for c in eval_retrieval.CATEGORIES} == {
            "narrative": 5, "numeric": 5, "mixed": 5,
        }
        assert len({q.id for q in qs}) == 15
        assert all(q.expected_pages for q in qs)

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            ("- {id: a, category: narrative, question: q, expected_pages: [1]}\n" * 2, "duplicate id"),
            ("- {id: a, category: trivia, question: q, expected_pages: [1]}", "category"),
            ("- {id: a, category: mixed, question: '', expected_pages: [1]}", "empty question"),
            ("- {id: a, category: mixed, question: q, expected_pages: [0]}", "positive PDF page"),
            ("- {id: a, category: mixed, question: q, expected_pages: 52}", "positive PDF page"),
        ],
    )
    def test_invalid_files(self, tmp_path: Path, body: str, message: str) -> None:
        path = tmp_path / "q.yaml"
        path.write_text("questions:\n" + "\n".join("  " + line for line in body.splitlines()))
        with pytest.raises(ValueError, match=message):
            load_questions(path)


class TestMain:
    @pytest.fixture(autouse=True)
    def no_live_search(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(query: str, k: int) -> list[int]:
            raise AssertionError("live search must not be called")

        monkeypatch.setattr(eval_retrieval, "live_search", fail)

    def test_check_makes_no_calls(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert eval_retrieval.main(["--check"]) == 0
        out = capsys.readouterr().out
        assert "15 questions (narrative=5, numeric=5, mixed=5); 15-30 embedding calls on the PAID key, 0 LLM calls." in out

    def test_declined_confirmation_makes_no_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda prompt: "n")
        assert eval_retrieval.main([]) == 1

    def test_only_verified_and_no_rerun_change_the_plan(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert eval_retrieval.main(["--check", "--only-verified", "--no-rerun"]) == 0
        assert "5 questions (narrative=0, numeric=5, mixed=0); 5-5 embedding calls" in capsys.readouterr().out

    def test_live_run_uses_search_and_prints_report(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(eval_retrieval, "live_search", lambda q, k: [52, 180, 102, 100, 6])
        assert eval_retrieval.main(["--yes"]) == 0
        out = capsys.readouterr().out
        assert "| t1 | numeric | yes | [52] | hit@1 [52, 180, 102, 100, 6] | - |" in out

    def test_bad_file_exits_1(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert eval_retrieval.main(["--check", "--questions", str(tmp_path / "missing.yaml")]) == 1
        assert "Error:" in capsys.readouterr().err
