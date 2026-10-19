"""Unit tests for validate_tables.py using synthetic tables (no PDF needed)."""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
import pytest

from extract_tables import ExtractedTable, SkippedLine, normalize_player_name
from validate_tables import (
    Check,
    check_all_nba,
    check_all_nba_by_season,
    check_draft,
    check_finals,
    cross_check_all_nba,
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
    df["player_search"] = df["player"].map(normalize_player_name)
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
    df["player_search"] = df["player"].map(normalize_player_name)
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


def _letters(n: int) -> str:
    """0 -> 'Aaa', 1 -> 'Aab', ...: unique capitalised surnames with no digits."""
    chars = []
    for _ in range(3):
        n, r = divmod(n, 26)
        chars.append(chr(ord("a") + r))
    return "".join(reversed(chars)).capitalize()


def _season_pair() -> tuple[ExtractedTable, dict[int, str], ExtractedTable, dict[int, str]]:
    """Consistent synthetic all_nba_by_season (80 seasons, 1951-52 tie) and matching all_nba_by_player."""
    rows: list[dict[str, object]] = []
    labels: list[str] = []
    n = 0
    for end in range(1947, 2027):
        season = f"{end - 1}-{end % 100:02d}"
        for level in ["First", "Second"] + (["Third"] if end >= 1989 else []):
            labels.append(level.upper())
            size = 6 if (season, level) == ("1951-52", "First") else 5
            for i in range(size):
                rows.append({
                    "season": season, "season_end_year": end, "team_level": level,
                    "player": f"Player {_letters(n)}", "team": "Tri-Cities",
                    "is_tie": int(size == 6 and i >= 4), "source_page": 104,
                })
                n += 1
    season_df = pd.DataFrame(rows)
    season_df["player_search"] = season_df["player"].map(normalize_player_name)
    season_text = " ".join(labels) + "\n" + " ".join(f"{r['player']}, {r['team']}" for r in rows)

    level_cols = {"First": "first_team", "Second": "second_team", "Third": "third_team"}
    player_df = pd.DataFrame([
        {"player": r["player"], "total": 1, "source_page": 111,
         **{c: int(r["team_level"] == lvl) for lvl, c in level_cols.items()}}
        for r in rows
    ])
    player_df["player_search"] = player_df["player"].map(normalize_player_name)
    player_text = "\n".join(
        f"{r.player} 1 {r.first_team} {r.second_team} {r.third_team}" for r in player_df.itertuples()
    )
    return ExtractedTable(season_df), {104: season_text}, ExtractedTable(player_df), {111: player_text}


class TestChecksPass:
    def test_clean_synthetic_tables_pass_everything(self) -> None:
        season, season_src, by_player, player_src = _season_pair()
        tables = {
            "draft_2026": _draft()[0], "all_nba_by_player": by_player, "nba_finals": _finals()[0],
            "all_nba_by_season": season,
        }
        source = {**_draft()[1], **player_src, **_finals()[1], **season_src}
        results = run_checks(tables, source)
        assert _failed(results) == []
        assert len(results) == 5 + 5 + 8 + 11 + 1

    def test_missing_table_is_a_failure(self) -> None:
        results = run_checks({}, {})
        assert [c.name for c in results] == ["table present in spec"] * 4
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


class TestAllNbaBySeasonChecks:
    @staticmethod
    def _failed_for(t: ExtractedTable, src: dict[int, str]) -> list[str]:
        return _failed(check_all_nba_by_season(t, src))

    def test_clean_table_passes(self) -> None:
        t, src, _, _ = _season_pair()
        assert self._failed_for(t, src) == []

    def test_missing_season_is_a_gap(self) -> None:
        t, src, _, _ = _season_pair()
        t.df = t.df[t.df["season"] != "1999-00"]
        check = next(c for c in check_all_nba_by_season(t, src) if c.name.startswith("seasons are exactly"))
        assert not check.passed
        assert "gaps=['1999-00']" in check.detail

    def test_wrong_season_end_year(self) -> None:
        t, src, _, _ = _season_pair()
        t.df.loc[t.df["season"] == "1999-00", "season_end_year"] = 1900
        assert "season_end_year is the season's ending year" in self._failed_for(t, src)

    def test_unexpected_group_size_is_listed(self) -> None:
        t, src, _, _ = _season_pair()
        t.df = t.df.drop(t.df.index[(t.df["season"] == "1996-97") & (t.df["team_level"] == "Third")][:1])
        check = next(c for c in check_all_nba_by_season(t, src) if c.name.startswith("5 players per"))
        assert not check.passed
        assert "('1996-97', 'Third'): 4" in check.detail and "('1951-52', 'First'): 6" in check.detail

    def test_tie_outside_1951_52_fails(self) -> None:
        t, src, _, _ = _season_pair()
        t.df.loc[t.df.index[0], "is_tie"] = 1
        assert "is_tie only on the 1951-52 First Team" in self._failed_for(t, src)

    def test_third_team_before_1988_89_fails(self) -> None:
        t, src, _, _ = _season_pair()
        t.df.loc[(t.df["season"] == "1987-88") & (t.df["team_level"] == "Second"), "team_level"] = "Third"
        failed = self._failed_for(t, src)
        assert "First+Second before 1988-89, plus Third from 1988-89" in failed
        assert "FIRST/SECOND/THIRD labels == level counts in page text" in failed

    @pytest.mark.parametrize(
        ("column", "value"),
        [("player", "Jordan"), ("team", "Phila-"), ("player", "Michael Jor dan,"), ("team", "chicago")],
    )
    def test_truncated_names_fail(self, column: str, value: str) -> None:
        t, src, _, _ = _season_pair()
        t.df.loc[t.df.index[10], column] = value
        assert "no truncated player or team names" in self._failed_for(t, src)

    def test_missing_team_fails(self) -> None:
        t, src, _, _ = _season_pair()
        t.df.loc[t.df.index[3], "team"] = ""
        assert "every row has a player and a team" in self._failed_for(t, src)

    def test_duplicate_player_in_season_fails(self) -> None:
        t, src, _, _ = _season_pair()
        first = t.df.index[0]
        t.df.loc[t.df.index[1], ["player", "player_search"]] = t.df.loc[first, ["player", "player_search"]].tolist()
        assert "no player twice in the same season" in self._failed_for(t, src)

    def test_row_count_must_match_source_entries(self) -> None:
        t, src, _, _ = _season_pair()
        src[104] += " Extra Player, Boston"
        assert "rows == 'Player, Team' entries in page text" in self._failed_for(t, src)

    @pytest.mark.parametrize("text", ["Tim Hardaway, Miami", "1996-97", "THIRD"])
    def test_data_like_skip_fails(self, text: str) -> None:
        t, src, _, _ = _season_pair()
        t.skipped = [SkippedLine(107, text, "not 'Player, Team'")]
        assert self._failed_for(t, src) == ["no data-like line skipped"]

    def test_title_skip_is_fine(self) -> None:
        t, src, _, _ = _season_pair()
        t.skipped = [SkippedLine(104, "ALL-NBA TEAM", "outside a season block")]
        assert self._failed_for(t, src) == []


class TestCrossCheck:
    def test_consistent_tables_match(self) -> None:
        season, _, by_player, _ = _season_pair()
        [check] = cross_check_all_nba(season.df, by_player.df)
        assert check.passed and check.detail == "0 mismatches"

    def test_every_mismatch_is_listed(self) -> None:
        season, _, by_player, _ = _season_pair()
        by_player.df.loc[0, ["first_team", "second_team"]] = [0, 1]          # level moved
        by_player.df = by_player.df.drop(index=5)                            # player missing from by_player
        [check] = cross_check_all_nba(season.df, by_player.df)
        assert not check.passed
        assert check.detail.startswith("2 mismatches: ")
        assert "player aaa: by_season 1st/2nd/3rd=[1, 0, 0] by_player=[0, 1, 0]" in check.detail
        # Row 5 is the first entry of the 1946-47 Second Team (rows 0-4 are the First Team).
        assert "player aaf: by_season 1st/2nd/3rd=[0, 1, 0] by_player=[0, 0, 0]" in check.detail

    def test_names_match_after_normalization(self) -> None:
        season = pd.DataFrame({"player_search": [normalize_player_name("Shaquille O’Neal")] * 2,
                               "team_level": ["First", "Third"]})
        by_player = pd.DataFrame({"player_search": [normalize_player_name("Shaquille O'Neal")],
                                  "first_team": [1], "second_team": [0], "third_team": [1]})
        assert cross_check_all_nba(season, by_player)[0].passed

    def test_cross_check_skipped_when_a_table_is_missing(self) -> None:
        season, season_src, _, _ = _season_pair()
        results = run_checks({"all_nba_by_season": season}, season_src)
        assert not any(c.name.startswith("matches all_nba_by_player") for c in results)

    def test_source_line_for_season_row(self) -> None:
        src = {107: "Tim Hardaway, Miami Patrick Ewing, New York Vin Baker, Milwaukee"}
        row = {"player": "Patrick Ewing", "team": "New York", "source_page": 107}
        assert source_line_for("all_nba_by_season", row, src) == src[107]
