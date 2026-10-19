"""Unit tests for validate_tables.py using synthetic tables (no PDF needed)."""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
import pytest

from extract_tables import ExtractedTable, SkippedLine
from validate_tables import (
    Check,
    check_all_nba,
    check_draft,
    check_finals,
    run_checks,
    source_line_for,
)


def _failed(checks: list[Check]) -> list[str]:
    return [c.name for c in checks if not c.passed]


def _draft() -> tuple[ExtractedTable, dict[int, str]]:
    df = pd.DataFrame({
        "pick": list(range(1, 61)),
        "player": [f"Player {i}" for i in range(1, 61)],
        "drafting_team": ["Team"] * 60,
        "before_nba": ["School"] * 60,
        "country": ["USA"] * 60,
        "source_page": [52] * 60,
    })
    text = "\n".join(f"{i}. Player {i} Team School USA" for i in range(1, 61))
    return ExtractedTable(df), {52: text}


def _all_nba() -> tuple[ExtractedTable, dict[int, str]]:
    df = pd.DataFrame({
        "player": ["LeBron James", "Amar’e Stoudemire", "Joe Johnson"],
        "total": [21, 5, 1],
        "first_team": [13, 1, 0],
        "second_team": [4, 4, 0],
        "third_team": [4, 0, 1],
        "source_page": [111, 111, 112],
    })
    source = {111: "LeBron James 21 13 4 4 Amar’e Stoudemire 5 1 4 0", 112: "Joe Johnson 1 0 0 1"}
    return ExtractedTable(df), source


def _finals() -> tuple[ExtractedTable, dict[int, str]]:
    years = list(range(1947, 2027))
    df = pd.DataFrame({
        "year": years,
        "winner_team": ["Boston"] * 80,
        "loser_team": ["LA Lakers"] * 80,
        "winner_games": [4] * 80,
        "loser_games": [2] * 80,
        "winner_best_record": [1] + [0] * 79,
        "loser_best_record": [0] * 80,
        "source_page": [179] * 80,
    })
    lines = [f"{y} June 1-9 Boston (C) LA Lakers (D) 4-2" for y in years]
    lines[0] = "1947 June 1-9 *Boston (C) LA Lakers (D) 4-2"
    return ExtractedTable(df), {179: "\n".join(lines)}


class TestChecksPass:
    def test_clean_synthetic_tables_pass_everything(self) -> None:
        tables = {"draft_2026": _draft()[0], "all_nba_by_player": _all_nba()[0], "nba_finals": _finals()[0]}
        source = {**_draft()[1], **_all_nba()[1], **_finals()[1]}
        results = run_checks(tables, source)
        assert _failed(results) == []
        assert len(results) == 18

    def test_missing_table_is_a_failure(self) -> None:
        results = run_checks({}, {})
        assert [c.name for c in results] == ["table present in spec"] * 3
        assert not any(c.passed for c in results)


    def test_missing_columns_fail_instead_of_crashing(self) -> None:
        t, src = _finals()
        t.df = t.df.drop(columns=["winner_best_record"])
        results = run_checks({"nba_finals": t}, src)
        finals = [c for c in results if c.table == "nba_finals"]
        assert [(c.name, c.passed) for c in finals] == [("required columns present", False)]
        assert "winner_best_record" in finals[0].detail


class TestDraftChecks:
    def test_gap_in_picks_fails(self) -> None:
        t, src = _draft()
        t.df = t.df[t.df["pick"] != 17]
        failed = _failed(check_draft(t, src))
        assert "row count == 60" in failed
        assert "picks are exactly 1..60" in failed
        assert "rows == numbered lines in page text" in failed

    def test_empty_field_and_data_like_skip_fail(self) -> None:
        t, src = _draft()
        t.df.loc[3, "country"] = ""
        t.skipped = [SkippedLine(52, "61. Ghost Player Team School USA", "words outside every column")]
        failed = _failed(check_draft(t, src))
        assert failed == ["no empty text fields", "no data-like line skipped"]

    def test_header_skip_is_not_data_like(self) -> None:
        t, src = _draft()
        t.skipped = [SkippedLine(52, "Pick Player Drafting Team Before NBA Country", "no leading pick number")]
        assert _failed(check_draft(t, src)) == []


class TestAllNbaChecks:
    def test_bad_sum_fails(self) -> None:
        t, src = _all_nba()
        t.df.loc[0, "third_team"] = 5
        check = next(c for c in check_all_nba(t, src) if not c.passed)
        assert check.name == "total == first+second+third on every row"
        assert "LeBron James" in check.detail

    def test_duplicate_and_dropped_player_fail(self) -> None:
        t, src = _all_nba()
        t.df.loc[2, "player"] = "LeBron James"
        failed = _failed(check_all_nba(t, src))
        assert "no duplicate players" in failed
        assert "same player names as page text" in failed

    def test_skipped_data_line_fails(self) -> None:
        t, src = _all_nba()
        t.skipped = [SkippedLine(111, "Mystery Player 3 1 1 1", "not a row")]
        assert _failed(check_all_nba(t, src)) == ["no data-like line skipped"]


class TestFinalsChecks:
    def test_missing_year_fails(self) -> None:
        t, src = _finals()
        t.df = t.df[t.df["year"] != 1990]
        failed = _failed(check_finals(t, src))
        assert "row count == 80" in failed
        assert "years are exactly 1947..2026 in order" in failed

    def test_winner_equals_loser_fails(self) -> None:
        t, src = _finals()
        t.df.loc[5, "loser_team"] = "Boston"
        assert _failed(check_finals(t, src)) == ["winner != loser on every row"]

    @pytest.mark.parametrize("team", ["11 LA Lakers", "*Boston", ""])
    def test_bad_team_name_fails(self, team: str) -> None:
        t, src = _finals()
        t.df.loc[73, "winner_team"] = team
        assert "team names have no digits/asterisks/empties" in _failed(check_finals(t, src))

    def test_lost_asterisk_fails(self) -> None:
        t, src = _finals()
        t.df.loc[0, "winner_best_record"] = 0
        assert _failed(check_finals(t, src)) == ["best-record flags == '*' count in page text"]


class TestSourceLine:
    def test_finds_player_inside_multi_block_line(self) -> None:
        _, src = _all_nba()
        row = {"player": "Amar’e Stoudemire", "source_page": 111}
        assert source_line_for("all_nba_by_player", row, src).startswith("LeBron James 21")

    def test_pick_prefix_does_not_match_longer_pick(self) -> None:
        _, src = _draft()
        assert source_line_for("draft_2026", {"pick": 1, "source_page": 52}, src) == "1. Player 1 Team School USA"
