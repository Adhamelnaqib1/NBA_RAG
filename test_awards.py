"""Offline tests for the award-list extraction (synthetic pages) and the awards checks."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd  # type: ignore[import-untyped]
import pytest

import extract_tables
import validate_tables
from extract_tables import (
    ExtractedTable,
    SkippedLine,
    award_end_year,
    normalize_heading,
    title_case_heading,
)

CHAR_W = 2.5


def word(text: str, x0: float, top: float, bold: bool = False, size: float = 7.0) -> dict[str, Any]:
    return {"text": text, "x0": x0, "x1": x0 + CHAR_W * len(text), "top": top, "bottom": top + size,
            "fontname": "MinionPro-Bold" if bold else "MinionPro-Regular", "size": size}


def line(text: str, x0: float, top: float, **kw: Any) -> list[dict[str, Any]]:
    out, x = [], x0
    for token in text.split():
        out.append(word(token, x, top, **kw))
        x += CHAR_W * len(token) + 2
    return out


def page(sections: list[tuple[str, list[list[str]]]], subtitle: bool = True) -> MagicMock:
    """Each section: a centered bold heading, an optional 'Winner receives' line spanning the gutters,
    then three columns of entries (x = 30, 190, 350)."""
    words: list[dict[str, Any]] = []
    top = 30.0
    for heading, columns in sections:
        words += line(heading, 120, top, bold=True, size=10.0)
        top += 12
        if subtitle:
            words += line("Winner receives The Very Long Named Trophy Of The Year", 100, top)
            top += 10
        for ci, entries in enumerate(columns):
            for ri, entry in enumerate(entries):
                words += line(entry, 30 + 160 * ci, top + 7 * ri)
        top += 7 * max(len(c) for c in columns) + 15
    pg = MagicMock()
    pg.extract_words.return_value = words
    return pg


def extract(pg: MagicMock) -> tuple[list[dict[str, Any]], list[SkippedLine]]:
    return extract_tables._extract_award_blocks(
        pg, 102, [20, 625], 12, ["AWARD WINNERS"], ["receives The ", "^Administered"]
    )


class TestHelpers:
    @pytest.mark.parametrize(
        ("raw", "titled"),
        [("NBA FINALS MOST V ALUABLE PLAYER", "NBA Finals Most Valuable Player"),
         ("NBA ALL-STAR GAME MOST VALUABLE PLAYER", "NBA All-Star Game Most Valuable Player"),
         ("J. WALTER KENNEDY CITIZENSHIP AWARD", "J. Walter Kennedy Citizenship Award"),
         ("NBA COACH OF THE YEAR", "NBA Coach of the Year"),
         ("NBA EASTERN CONFERENCE FINALS MVP", "NBA Eastern Conference Finals MVP")],
    )
    def test_title_case_heading(self, raw: str, titled: str) -> None:
        assert title_case_heading(raw) == titled

    def test_normalize_heading_joins_split_letters(self) -> None:
        assert normalize_heading("MOST V ALUABLE  PLAYER") == "MOST VALUABLE PLAYER"

    @pytest.mark.parametrize(("label", "year"), [("2025-26", 2026), ("1999-00", 2000), ("2026", 2026), ("1969", 1969)])
    def test_award_end_year(self, label: str, year: int) -> None:
        assert award_end_year(label) == year


class TestExtraction:
    def test_heading_stays_with_its_own_list_by_position(self) -> None:
        pg = page([
            ("NBA FINALS MOST V ALUABLE PLAYER", [["1969 — Jerry West, LA Lakers", "1970 — Willis Reed, New York"],
                                                 ["2021 — Giannis Antetokounmpo, Milwaukee"], ["2026 — Jalen Brunson, New York"]]),
            ("NBA ALL-STAR GAME MOST VALUABLE PLAYER", [["1951 — Ed Macauley, Boston"], ["2021 — Giannis Antetokounmpo, Milwaukee"],
                                                       ["2026 — Anthony Edwards, Minnesota"]]),
        ])
        rows, skipped = extract(pg)
        assert skipped == []
        assert [(r["award"], r["season_label"], r["player"]) for r in rows] == [
            ("NBA Finals Most Valuable Player", "1969", "Jerry West"),
            ("NBA Finals Most Valuable Player", "1970", "Willis Reed"),
            ("NBA Finals Most Valuable Player", "2021", "Giannis Antetokounmpo"),
            ("NBA Finals Most Valuable Player", "2026", "Jalen Brunson"),
            ("NBA All-Star Game Most Valuable Player", "1951", "Ed Macauley"),
            ("NBA All-Star Game Most Valuable Player", "2021", "Giannis Antetokounmpo"),
            ("NBA All-Star Game Most Valuable Player", "2026", "Anthony Edwards"),
        ]

    def test_ties_co_recipients_and_not_awarded(self) -> None:
        pg = page([
            ("NBA ROOKIE OF THE YEAR", [["1970-71 — Dave Cowens, Boston (tie)", "Geoff Petrie, Portland (tie)"],
                                        ["2019-20 — Harrison Barnes (Sacramento)", "Jaylen Brown (Boston)"],
                                        ["2020-21 — N/A (pandemic)", "2021-22 — Scottie Barnes, Toronto"]]),
        ])
        rows, skipped = extract(pg)
        assert [(r["season_label"], r["player"], r["team"], r["is_tie"]) for r in rows] == [
            ("1970-71", "Dave Cowens", "Boston", 1), ("1970-71", "Geoff Petrie", "Portland", 1),
            ("2019-20", "Harrison Barnes", "Sacramento", 0), ("2019-20", "Jaylen Brown", "Boston", 0),
            ("2021-22", "Scottie Barnes", "Toronto", 0),
        ]
        assert [(s.text, s.reason) for s in skipped] == [
            ("2020-21 — N/A (pandemic)", "not awarded in 2020-21: N/A (pandemic)")]
        assert rows[0]["season_end_year"] == 1971

    def test_subtitle_spanning_the_gutters_is_dropped_before_columns_are_found(self) -> None:
        pg = page([("NBA MOST VALUABLE PLAYER", [["1955-56 — Bob Pettit, St. Louis"], ["1979-80 — Kareem Abdul-Jabbar, LA Lakers"],
                                                  ["2003-04 — Kevin Garnett, Minnesota"]])])
        rows, _ = extract(pg)
        assert len(rows) == 3 and rows[2]["player"] == "Kevin Garnett"

    def test_malformed_and_orphan_lines_are_skipped(self) -> None:
        pg = page([("NBA SPORTSMANSHIP AWARD", [["Orphan Line, Somewhere", "1995-96 — Joe Dumars, Detroit", "1996-97 — Not a winner line"],
                                                  ["1997-98 — Avery Johnson, San Antonio"], ["1998-99 — Hersey Hawkins, Seattle"]])])
        rows, skipped = extract(pg)
        assert len(rows) == 3
        assert [s.reason for s in skipped] == ["no year before this line in NBA Sportsmanship Award",
                                               "not 'Player, Team' in NBA Sportsmanship Award 1996-97"]


def awards_df(rows: list[tuple[str, str, str, str, int]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["award", "season_label", "player", "team", "is_tie"])
    df["season_end_year"] = df["season_label"].map(award_end_year)
    df["player_search"] = df["player"].map(extract_tables.normalize_player_name)
    df["source_page"] = 102
    return df


def source_for(df: pd.DataFrame) -> dict[int, str]:
    lines: list[str] = []
    for award, group in df.groupby("award", sort=False):
        lines.append(award.upper())
        lines += [f"{r.season_label} — {r.player}, {r.team}" for r in group.itertuples()]
    return {p: ("\n".join(lines) if p == 102 else "") for p in validate_tables.SOURCE_PAGES["awards"]}


FMVP = "NBA Finals Most Valuable Player"
MVP = "NBA Most Valuable Player"


class TestChecks:
    def clean(self) -> pd.DataFrame:
        return awards_df([
            (FMVP, "1969", "Jerry West", "LA Lakers", 0), (FMVP, "1970", "Willis Reed", "New York", 0),
            (MVP, "1969-70", "Willis Reed", "New York", 0), (MVP, "1970-71", "Kareem Abdul-Jabbar", "Milwaukee", 0),
        ])

    def failed(self, df: pd.DataFrame, skipped: list[SkippedLine] | None = None) -> list[str]:
        checks = validate_tables.check_awards(ExtractedTable(df, skipped or []), source_for(df))
        return [c.name for c in checks if not c.passed]

    def test_clean_table_passes(self) -> None:
        assert self.failed(self.clean()) == []

    def test_gap_fails_unless_documented(self) -> None:
        df = self.clean()
        df.loc[len(df)] = df.iloc[3].copy()
        df.loc[len(df) - 1, ["season_label", "season_end_year"]] = ["1972-73", 1973]
        assert "years consecutive per award (gaps only where documented)" in self.failed(df)

    def test_row_count_must_match_source(self) -> None:
        df = self.clean()
        src = source_for(df)
        src[102] += "\n1971 — Extra Player, Boston"
        checks = validate_tables.check_awards(ExtractedTable(df), src)
        assert "rows == entry lines in page text, per award" in [c.name for c in checks if not c.passed]

    def test_mixed_year_format_fails(self) -> None:
        df = self.clean()
        df.loc[0, "season_label"] = "1968-69"
        assert "heading matches content: calendar years only for Finals/All-Star MVPs, seasons otherwise" in self.failed(df)

    def test_missing_team_and_truncated_name_fail(self) -> None:
        df = self.clean()
        df.loc[1, "team"] = ""
        df.loc[2, "player"] = "Reed"
        failed = self.failed(df)
        assert "every row has a team" in failed and "no truncated player or team names" in failed

    def test_unexplained_multi_winner_year_fails(self) -> None:
        df = self.clean()
        df.loc[len(df)] = [MVP, "1970-71", "Someone Else", "Boston", 0, 1971, "someone else", 102]
        assert "several winners in a year only as ties or documented co-recipients" in self.failed(df)

    def test_data_like_skip_fails_but_not_awarded_is_fine(self) -> None:
        assert self.failed(self.clean(), [SkippedLine(103, "2020-21 — N/A (pandemic)", "not awarded in 2020-21")]) == []
        assert self.failed(self.clean(), [SkippedLine(102, "1971 — Lost Row, Boston", "not 'Player, Team'")]) == [
            "no data-like line skipped"]


class TestCrossChecks:
    finals = pd.DataFrame({"year": [1969, 1970], "winner_team": ["Boston", "New York"]})
    by_season = pd.DataFrame({"season": ["1969-70", "1970-71"], "team_level": ["First", "Second"],
                              "player_search": ["willis reed", "kareem abdul-jabbar"]})

    def test_finals_mvp_from_losing_team_only_where_documented(self) -> None:
        df = TestChecks().clean()
        [fmvp, _] = validate_tables.cross_check_awards(df, self.finals, self.by_season)
        assert fmvp.passed and "1969: 'Jerry West, LA Lakers vs winner Boston'" in fmvp.detail
        df.loc[1, "team"] = "LA Lakers"
        [fmvp, _] = validate_tables.cross_check_awards(df, self.finals, self.by_season)
        assert not fmvp.passed and "undocumented=[1970]" in fmvp.detail

    def test_mvp_not_on_first_team_is_a_warning(self) -> None:
        [_, mvp] = validate_tables.cross_check_awards(TestChecks().clean(), self.finals, self.by_season)
        assert mvp.passed and mvp.warning
        assert "1970-71 Kareem Abdul-Jabbar: printed on the Second Team that season" in mvp.detail
        assert validate_tables.status(mvp) == "WARN"
