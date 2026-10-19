"""Dry-run validation for the pilot stat tables (draft_2026, all_nba_by_player, nba_finals).

Extracts the tables with extract_tables.load_tables_from_spec (no database write), checks them
against invariants and against counts parsed independently from the raw page text, and prints
PASS/FAIL lines plus random sample rows next to the source line they came from.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pdfplumber

from extract_tables import (
    ExtractedTable,
    SkippedLine,
    load_tables_from_spec,
    normalize_heading,
)

logger = logging.getLogger(__name__)

SOURCE_PAGES: dict[str, list[int]] = {
    "draft_2026": [52],
    "all_nba_by_player": [111, 112],
    "nba_finals": [179, 180],
    "all_nba_by_season": [104, 105, 106, 107, 108, 109],
    "awards": [100, 101, 102, 103],
}

# Patterns over raw page text, independent of the extractor's bboxes, x-splits and line clustering.
DRAFT_SOURCE_RE = re.compile(r"^(\d+)\. ")
ALL_NBA_SOURCE_RE = re.compile(r"([A-Z][^\d]*?) (\d+) (\d+) (\d+) (\d+)(?= |$)")
FINALS_SOURCE_RE = re.compile(r"^(\d{4}) ")

# A skipped line that matches one of these looked like data, so skipping it lost a row.
DATA_LIKE: dict[str, re.Pattern[str]] = {
    "awards": re.compile(r"\b\d{4}(?:-\d{2})?\s+—\s+|\(tie\)|, [A-Z]"),
    "draft_2026": re.compile(r"^\d+\.?\s"),
    "all_nba_by_player": re.compile(r"\d+ \d+ \d+ \d+$"),
    "nba_finals": re.compile(r"^\d{4}\b"),
    "all_nba_by_season": re.compile(r", [A-Z]|^\d{4}-\d{2}$|^(FIRST|SECOND|THIRD)$"),
}


@dataclass(frozen=True)
class Check:
    """One named validation result."""

    table: str
    name: str
    passed: bool
    detail: str = ""
    warning: bool = False
    """A warning is reported (WARN) but never blocks the load."""


def status(c: Check) -> str:
    return "WARN" if c.warning else "PASS" if c.passed else "FAIL"


def _source_lines(source_text: dict[int, str], pages: list[int]) -> list[str]:
    return [line.strip() for p in pages for line in source_text.get(p, "").splitlines() if line.strip()]


def _data_like_skips(table: str, skipped: list[SkippedLine]) -> list[SkippedLine]:
    pattern = DATA_LIKE[table]
    return [s for s in skipped if pattern.search(s.text)]


def _check(table: str, name: str, passed: bool, detail: str = "") -> Check:
    return Check(table=table, name=name, passed=bool(passed), detail=detail)


def check_draft(t: ExtractedTable, source_text: dict[int, str]) -> list[Check]:
    """Draft: 60 rows, picks 1..60 with no gaps, no empty fields, matches the source line count."""
    name = "draft_2026"
    df = t.df
    source = [ln for ln in _source_lines(source_text, SOURCE_PAGES[name]) if DRAFT_SOURCE_RE.match(ln)]
    picks = sorted(df["pick"].tolist()) if "pick" in df else []
    missing = sorted(set(range(1, 61)) - set(picks))
    text_cols = ["player", "drafting_team", "before_nba", "country"]
    empty = df[(df[text_cols] == "").any(axis=1)] if not df.empty else df
    data_skips = _data_like_skips(name, t.skipped)
    return [
        _check(name, "row count == 60", len(df) == 60, f"rows={len(df)}"),
        _check(name, "picks are exactly 1..60", picks == list(range(1, 61)), f"missing={missing}"),
        _check(name, "no empty text fields", empty.empty, f"picks with empties={empty.get('pick', pd.Series()).tolist()}"),
        _check(name, "rows == numbered lines in page text", len(df) == len(source), f"rows={len(df)} source={len(source)}"),
        _check(name, "no data-like line skipped", not data_skips, f"{[s.text for s in data_skips]}"),
    ]


def check_all_nba(t: ExtractedTable, source_text: dict[int, str]) -> list[Check]:
    """All-NBA: total == 1st+2nd+3rd on every row, unique players, same players as the page text."""
    name = "all_nba_by_player"
    df = t.df
    source_players = [
        m.group(1).strip()
        for ln in _source_lines(source_text, SOURCE_PAGES[name])
        for m in ALL_NBA_SOURCE_RE.finditer(ln)
    ]
    bad_sum = df[df["total"] != df["first_team"] + df["second_team"] + df["third_team"]] if not df.empty else df
    dupes = sorted(set(df.loc[df["player"].duplicated(keep=False), "player"])) if not df.empty else []
    extracted = set(df["player"]) if not df.empty else set()
    only_source = sorted(set(source_players) - extracted)
    only_extracted = sorted(extracted - set(source_players))
    data_skips = _data_like_skips(name, t.skipped)
    return [
        _check(name, "total == first+second+third on every row", bad_sum.empty,
               f"bad rows={bad_sum.get('player', pd.Series()).tolist()}"),
        _check(name, "no duplicate players", not dupes, f"dupes={dupes}"),
        _check(name, "rows == player entries in page text", len(df) == len(source_players),
               f"rows={len(df)} source={len(source_players)}"),
        _check(name, "same player names as page text", not only_source and not only_extracted,
               f"only in source={only_source} only extracted={only_extracted}"),
        _check(name, "no data-like line skipped", not data_skips, f"{[s.text for s in data_skips]}"),
    ]


def check_finals(t: ExtractedTable, source_text: dict[int, str]) -> list[Check]:
    """Finals: 80 rows, years 1947..2026 consecutive, winner != loser, asterisk flags match the page."""
    name = "nba_finals"
    df = t.df
    source = [ln for ln in _source_lines(source_text, SOURCE_PAGES[name]) if FINALS_SOURCE_RE.match(ln)]
    source_stars = sum(ln.count("*") for ln in source)
    years = df["year"].tolist() if "year" in df else []
    same_team = df[df["winner_team"] == df["loser_team"]] if not df.empty else df
    bad_games = df[(df["winner_games"] != 4) | ~df["loser_games"].between(0, 3)] if not df.empty else df
    team_cols = ["winner_team", "loser_team"]
    bad_teams = df[df[team_cols].apply(lambda c: c.str.contains(r"[\d*]") | (c == "")).any(axis=1)] if not df.empty else df
    flags = int(df["winner_best_record"].sum() + df["loser_best_record"].sum()) if "winner_best_record" in df else -1
    data_skips = _data_like_skips(name, t.skipped)
    return [
        _check(name, "row count == 80", len(df) == 80, f"rows={len(df)}"),
        _check(name, "years are exactly 1947..2026 in order", years == list(range(1947, 2027)),
               f"missing={sorted(set(range(1947, 2027)) - set(years))}"),
        _check(name, "winner != loser on every row", same_team.empty, f"years={same_team.get('year', pd.Series()).tolist()}"),
        _check(name, "games are 4-0..4-3", bad_games.empty, f"years={bad_games.get('year', pd.Series()).tolist()}"),
        _check(name, "team names have no digits/asterisks/empties", bad_teams.empty,
               f"years={bad_teams.get('year', pd.Series()).tolist()}"),
        _check(name, "best-record flags == '*' count in page text", flags == source_stars,
               f"flags={flags} source stars={source_stars}"),
        _check(name, "rows == year lines in page text", len(df) == len(source), f"rows={len(df)} source={len(source)}"),
        _check(name, "no data-like line skipped", not data_skips, f"{[s.text for s in data_skips]}"),
    ]


FIRST_SEASON_END, LAST_SEASON_END, THIRD_TEAM_FROM = 1947, 2026, 1989
ALLOWED_SIZE_EXCEPTIONS: dict[tuple[str, str], int] = {("1951-52", "First"): 6}
"""(season, team_level) groups allowed to differ from 5 players: the 1951-52 First Team tie."""
LEVEL_COLUMNS: dict[str, str] = {"First": "first_team", "Second": "second_team", "Third": "third_team"}


def _season_label(end_year: int) -> str:
    return f"{end_year - 1}-{end_year % 100:02d}"


def check_all_nba_by_season(t: ExtractedTable, source_text: dict[int, str]) -> list[Check]:
    """All-NBA by season: every season once and in order, 5 per team (one tie), 2 or 3 levels, clean names."""
    name = "all_nba_by_season"
    df = t.df
    page_text = "\n".join(source_text.get(p, "") for p in SOURCE_PAGES[name])

    expected_seasons = [_season_label(y) for y in range(FIRST_SEASON_END, LAST_SEASON_END + 1)]
    seasons = list(dict.fromkeys(df["season"]))
    gaps = sorted(set(expected_seasons) - set(seasons))
    extra = sorted(set(seasons) - set(expected_seasons))
    bad_end = df[df["season"].map(lambda x: int(x[:4]) + 1) != df["season_end_year"]]

    sizes = df.groupby(["season", "team_level"], sort=False).size()
    exceptions = {key: int(n) for key, n in sizes.items() if n != 5}
    ties = df[df["is_tie"] == 1]
    tie_groups = sorted({(r.season, r.team_level) for r in ties.itertuples()})

    wrong_levels: list[str] = []
    for season, group in df.groupby("season", sort=False):
        expected = ["First", "Second"] + (["Third"] if int(group["season_end_year"].iloc[0]) >= THIRD_TEAM_FROM else [])
        levels = list(dict.fromkeys(group["team_level"]))
        if levels != expected:
            wrong_levels.append(f"{season}: {levels}")
    source_labels = {lvl: len(re.findall(rf"\b{lvl.upper()}\b", page_text)) for lvl in LEVEL_COLUMNS}
    parsed_labels = {lvl: int(df.loc[df["team_level"] == lvl, "season"].nunique()) for lvl in LEVEL_COLUMNS}

    source_entries = page_text.count(", ")
    missing_fields = df[(df["player"].str.strip() == "") | (df["team"].str.strip() == "")]
    name_shape = re.compile(r"^[A-Z][\w.'’-]*( [\w.'’-]+)+$")
    team_shape = re.compile(r"^[A-Z][A-Za-z.'’/ -]*[A-Za-z.]$")
    truncated = df[~df["player"].str.match(name_shape) | ~df["team"].str.match(team_shape)]
    dupes = df[df.duplicated(["season", "player_search"], keep=False)]
    data_skips = _data_like_skips(name, t.skipped)

    def rows(frame: pd.DataFrame) -> list[str]:
        return [f"{r.season} {r.team_level}: {r.player}, {r.team}" for r in frame.itertuples()]

    return [
        _check(name, "seasons are exactly 1946-47..2025-26 in order", seasons == expected_seasons,
               f"seasons={len(seasons)} gaps={gaps} unexpected={extra}"),
        _check(name, "season_end_year is the season's ending year", bad_end.empty, f"{rows(bad_end)}"),
        _check(name, "5 players per (season, level); only 1951-52 First has 6", exceptions == ALLOWED_SIZE_EXCEPTIONS,
               f"exceptions={exceptions}"),
        _check(name, "is_tie only on the 1951-52 First Team", tie_groups == [("1951-52", "First")],
               f"tie rows={rows(ties)}"),
        _check(name, "First+Second before 1988-89, plus Third from 1988-89", not wrong_levels, f"{wrong_levels}"),
        _check(name, "FIRST/SECOND/THIRD labels == level counts in page text", parsed_labels == source_labels,
               f"parsed={parsed_labels} source={source_labels}"),
        _check(name, "rows == 'Player, Team' entries in page text", len(df) == source_entries,
               f"rows={len(df)} source={source_entries}"),
        _check(name, "every row has a player and a team", missing_fields.empty, f"{rows(missing_fields)}"),
        _check(name, "no truncated player or team names", truncated.empty, f"{rows(truncated)}"),
        _check(name, "no player twice in the same season", dupes.empty, f"{rows(dupes)}"),
        _check(name, "no data-like line skipped", not data_skips, f"{[(s.page, s.text) for s in data_skips]}"),
    ]


def cross_check_all_nba(by_season: pd.DataFrame, by_player: pd.DataFrame) -> list[Check]:
    """Per player and level, all_nba_by_season counts must equal all_nba_by_player's totals."""
    counts = (
        by_season.groupby(["player_search", "team_level"]).size().unstack(fill_value=0)
        .reindex(columns=list(LEVEL_COLUMNS), fill_value=0)
        .rename(columns=LEVEL_COLUMNS)
    )
    totals = by_player.set_index("player_search")[list(LEVEL_COLUMNS.values())]
    joined = counts.join(totals, how="outer", lsuffix="_season", rsuffix="_player").fillna(0).astype(int)
    mismatches: list[str] = []
    for player, r in joined.iterrows():
        season_counts = [int(r[f"{c}_season"]) for c in LEVEL_COLUMNS.values()]
        player_counts = [int(r[f"{c}_player"]) for c in LEVEL_COLUMNS.values()]
        if season_counts != player_counts:
            mismatches.append(f"{player}: by_season 1st/2nd/3rd={season_counts} by_player={player_counts}")
    return [
        _check("all_nba_by_season", "matches all_nba_by_player per player and level", not mismatches,
               f"{len(mismatches)} mismatches: " + "; ".join(mismatches) if mismatches else "0 mismatches"),
    ]


CALENDAR_YEAR_AWARDS: frozenset[str] = frozenset({
    "NBA Finals Most Valuable Player", "NBA Eastern Conference Finals MVP", "NBA Western Conference Finals MVP",
    "NBA All-Star Game Most Valuable Player",
})
AWARD_GAPS: dict[str, dict[int, str]] = {
    "NBA All-Star Game Most Valuable Player": {1999: "1999 is not in the guide's list (no All-Star Game that year)"},
    "J. Walter Kennedy Citizenship Award": {2021: "printed as N/A (pandemic)", 2022: "printed as N/A (pandemic)"},
}
"""Years inside an award's first..last range that the guide does not list, with the reason."""
AWARD_CO_RECIPIENTS: dict[tuple[str, str], str] = {
    ("Bob Lanier Community Assist Award", "2019-20"): "five recipients printed as 'Player (Team)', no tie marker",
}
FINALS_MVP_NOT_CHAMPION: dict[int, str] = {
    1969: "Jerry West, LA Lakers: the only Finals MVP from the losing team (Boston won 4-3, PDF p. 179)",
}
REGULAR_MVP = "NBA Most Valuable Player"
FINALS_MVP = "NBA Finals Most Valuable Player"
_PAREN_TEAM_TEXT = re.compile(r"[A-Z][\w.'’-]+(?: [A-Z][\w.'’-]+)* \([A-Z][A-Za-z.' ]+\)")


def award_source_counts(source_text: dict[int, str], awards: list[str]) -> dict[str, int]:
    """Rows each award should have, counted from the raw page text between its heading and the next one.

    Every winner is printed once as "Player, Team" (one ", "; no team or player name in the guide contains a
    comma) or as "Player (Team)"; "N/A (...)" years have neither. Counting these is independent of how the
    flattened text interleaves the columns."""
    headings = {a.upper(): a for a in awards}
    counts: dict[str, int] = dict.fromkeys(awards, 0)
    current: str | None = None
    segments: dict[str, list[str]] = {a: [] for a in awards}
    for page in SOURCE_PAGES["awards"]:
        for line in source_text.get(page, "").splitlines():
            printed = normalize_heading(line.strip())
            if printed.upper() in headings:
                current = headings[printed.upper()]
            elif printed.isupper() and len(printed) > 12 and "—" not in printed:
                current = None  # another all-caps heading (e.g. AWARD WINNERS)
            elif current is not None:
                segments[current].append(line)
    for award, lines in segments.items():
        text = "\n".join(lines)
        counts[award] = text.count(", ") + len(_PAREN_TEAM_TEXT.findall(text))
    return counts


def check_awards(t: ExtractedTable, source_text: dict[int, str]) -> list[Check]:
    """Awards: consecutive years (documented gaps), row counts vs page text, names, teams, heading/content match."""
    name = "awards"
    df = t.df
    gaps_found: dict[str, list[int]] = {}
    undocumented: list[str] = []
    format_errors: list[str] = []
    for award, group in df.groupby("award", sort=False):
        years = sorted(set(group["season_end_year"]))
        missing = sorted(set(range(years[0], years[-1] + 1)) - set(years))
        if missing:
            gaps_found[str(award)] = missing
        documented = AWARD_GAPS.get(str(award), {})
        undocumented += [f"{award} {y}" for y in missing if y not in documented]
        calendar = award in CALENDAR_YEAR_AWARDS
        bad = group[~group["season_label"].str.fullmatch(r"\d{4}" if calendar else r"\d{4}-\d{2}")]
        format_errors += [f"{award}: {lbl}" for lbl in bad["season_label"]]
    documented_gaps = {a: {y: AWARD_GAPS.get(a, {}).get(y, "UNDOCUMENTED") for y in ys} for a, ys in gaps_found.items()}

    awards = list(dict.fromkeys(df["award"]))
    source = award_source_counts(source_text, awards)
    parsed = df.groupby("award", sort=False).size().to_dict()
    count_mismatch = {a: (int(parsed.get(a, 0)), source[a]) for a in awards if int(parsed.get(a, 0)) != source[a]}

    multi = df.groupby(["award", "season_label"]).filter(lambda g: len(g) > 1)
    unexplained_multi = sorted({
        f"{r.award} {r.season_label}" for r in multi.itertuples()
        if not r.is_tie and (r.award, r.season_label) not in AWARD_CO_RECIPIENTS
    })
    name_shape = re.compile(r"^[A-Z][\w.'’-]*( [\w.'’-]+)+$")
    team_shape = re.compile(r"^[A-Z][A-Za-z.'’/ -]*[A-Za-z.]$")
    truncated = df[~df["player"].str.match(name_shape) | ~df["team"].str.match(team_shape)]
    no_team = df[df["team"].str.strip() == ""]
    heading_dupes = [a for a, n in df.drop_duplicates(["award", "source_page"]).groupby("award").size().items() if n > 1]
    data_skips = [s for s in _data_like_skips(name, t.skipped) if not s.reason.startswith("not awarded")]
    not_awarded = [f"p.{s.page}: {s.text}" for s in t.skipped if s.reason.startswith("not awarded")]

    def rows(frame: pd.DataFrame) -> list[str]:
        return [f"{r.award} {r.season_label}: {r.player}, {r.team}" for r in frame.itertuples()]

    return [
        _check(name, "years consecutive per award (gaps only where documented)", not undocumented,
               f"gaps={documented_gaps} undocumented={undocumented}"),
        _check(name, "rows == entry lines in page text, per award", not count_mismatch,
               f"mismatches (parsed, source)={count_mismatch}; total rows={len(df)}"),
        _check(name, "no truncated player or team names", truncated.empty, f"{rows(truncated)}"),
        _check(name, "every row has a team", no_team.empty, f"{rows(no_team)}"),
        _check(name, "several winners in a year only as ties or documented co-recipients", not unexplained_multi,
               f"unexplained={unexplained_multi}; ties={int(df['is_tie'].sum())} rows; "
               f"co-recipients={list(AWARD_CO_RECIPIENTS)}"),
        _check(name, "heading matches content: calendar years only for Finals/All-Star MVPs, seasons otherwise",
               not format_errors, f"{format_errors}"),
        _check(name, "each award heading appears once", not heading_dupes, f"{heading_dupes}"),
        _check(name, "no data-like line skipped", not data_skips,
               f"{[(s.page, s.text) for s in data_skips]}; not awarded (documented)={not_awarded}"),
    ]


def cross_check_awards(awards: pd.DataFrame, finals: pd.DataFrame, by_season: pd.DataFrame) -> list[Check]:
    """(a) Finals MVP team == nba_finals winner (documented exceptions only); (b) MVP on All-NBA First (warning)."""
    fmvp = awards[awards["award"] == FINALS_MVP].merge(finals, left_on="season_end_year", right_on="year", how="left")
    exceptions = {int(r.season_end_year): f"{r.player}, {r.team} vs winner {r.winner_team}"
                  for r in fmvp.itertuples() if r.team != r.winner_team}
    undocumented = sorted(set(exceptions) - set(FINALS_MVP_NOT_CHAMPION))
    first = by_season[by_season["team_level"] == "First"]
    first_keys = set(zip(first["season"], first["player_search"]))
    mvp = awards[awards["award"] == REGULAR_MVP]
    not_first: list[str] = []
    for r in mvp.itertuples():
        if (r.season_label, r.player_search) not in first_keys:
            level = by_season[(by_season["season"] == r.season_label) & (by_season["player_search"] == r.player_search)]
            where = f"{level['team_level'].iloc[0]} Team that season" if not level.empty else "not on any All-NBA team"
            not_first.append(f"{r.season_label} {r.player}: printed on the {where}")
    return [
        _check("awards", "Finals MVP team == nba_finals winner (documented exceptions only)", not undocumented,
               f"exceptions={exceptions}; documented={FINALS_MVP_NOT_CHAMPION}; undocumented={undocumented}"),
        Check("awards", "regular-season MVP is on the All-NBA First Team that season", True,
              f"{len(not_first)} exceptions: " + "; ".join(not_first) if not_first else "0 exceptions",
              warning=bool(not_first)),
    ]


REQUIRED_COLUMNS: dict[str, list[str]] = {
    "draft_2026": ["pick", "player", "player_search", "drafting_team", "before_nba", "country", "source_page"],
    "all_nba_by_player": [
        "player", "player_search", "total", "first_team", "second_team", "third_team", "source_page",
    ],
    "all_nba_by_season": [
        "season", "season_end_year", "team_level", "player", "player_search", "team", "is_tie", "source_page",
    ],
    "awards": ["award", "season_label", "season_end_year", "player", "player_search", "team", "is_tie", "source_page"],
    "nba_finals": [
        "year", "winner_team", "loser_team", "winner_games", "loser_games",
        "winner_best_record", "loser_best_record", "source_page",
    ],
}

CHECKS: dict[str, Callable[[ExtractedTable, dict[int, str]], list[Check]]] = {
    "draft_2026": check_draft,
    "all_nba_by_player": check_all_nba,
    "nba_finals": check_finals,
    "all_nba_by_season": check_all_nba_by_season,
    "awards": check_awards,
}


def run_checks(tables: dict[str, ExtractedTable], source_text: dict[int, str]) -> list[Check]:
    """Runs every pilot-table check; a pilot table missing from `tables` is itself a failure."""
    results: list[Check] = []
    for name, check_fn in CHECKS.items():
        if name not in tables:
            results.append(_check(name, "table present in spec", False, "missing"))
            continue
        missing = [c for c in REQUIRED_COLUMNS[name] if c not in tables[name].df.columns]
        if missing:
            results.append(_check(name, "required columns present", False, f"missing={missing}"))
            continue
        results.extend(check_fn(tables[name], source_text))

    usable = {r.table for r in results} - {r.table for r in results if r.name in _STRUCTURAL_CHECKS and not r.passed}
    if {"all_nba_by_season", "all_nba_by_player"} <= usable:
        results.extend(cross_check_all_nba(tables["all_nba_by_season"].df, tables["all_nba_by_player"].df))
    if {"awards", "nba_finals", "all_nba_by_season"} <= usable:
        results.extend(cross_check_awards(tables["awards"].df, tables["nba_finals"].df, tables["all_nba_by_season"].df))
    return results


_STRUCTURAL_CHECKS = frozenset({"table present in spec", "required columns present"})


def source_line_for(table: str, row: dict[str, Any], source_text: dict[int, str]) -> str:
    """Finds the raw page-text line a row came from (by pick, player name or year)."""
    lines = _source_lines(source_text, [int(row["source_page"])])
    if table == "draft_2026":
        prefix = f"{row['pick']}. "
        match = [ln for ln in lines if ln.startswith(prefix)]
    elif table == "all_nba_by_player":
        match = [ln for ln in lines if re.search(rf"(^| ){re.escape(row['player'])} \d", ln)]
    elif table == "all_nba_by_season":
        match = [ln for ln in lines if f"{row['player']}, {row['team']}" in ln]
    elif table == "awards":
        match = [ln for ln in lines if f"{row['season_label']} — {row['player']}" in ln]
        match = match or [ln for ln in lines if row["player"] in ln]
    else:
        match = [ln for ln in lines if ln.startswith(f"{row['year']} ")]
    return match[0] if match else "<no source line found>"


def read_source_text(pdf_path: str | Path) -> dict[int, str]:
    """Reads the full, uncropped text of every pilot page."""
    pages = sorted({p for ps in SOURCE_PAGES.values() for p in ps})
    with pdfplumber.open(Path(pdf_path).resolve()) as pdf:
        return {p: pdf.pages[p - 1].extract_text() or "" for p in pages}


def main() -> None:
    """CLI: dry-run extract, validate, and print samples. Never writes the database."""
    parser = argparse.ArgumentParser(description="Validate pilot tables extracted from the NBA Guide PDF.")
    parser.add_argument("--pdf", default="2026-27-NBA-Guide.pdf", help="Path to PDF file")
    parser.add_argument("--spec", default="tables.yaml", help="Path to tables.yaml")
    parser.add_argument("--samples", type=int, default=5, help="Random sample rows per table (default: 5)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for samples (default: 0)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log every skipped line as it happens")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    tables = load_tables_from_spec(pdf_path=args.pdf, spec_path=args.spec)
    source_text = read_source_text(args.pdf)

    for name, t in tables.items():
        print(f"\n=== {name}: {len(t.df)} rows, {len(t.skipped)} skipped lines")
        for s in t.skipped:
            print(f"  skipped p.{s.page} ({s.reason}): {s.text!r}")

    results = run_checks(tables, source_text)
    print("\n=== Checks")
    for c in results:
        print(f"{status(c)}  {c.table}: {c.name}  [{c.detail}]")

    for name, t in tables.items():
        if t.df.empty:
            continue
        print(f"\n=== {name}: {args.samples} random rows (seed={args.seed}) vs source text")
        sample = t.df.sample(n=min(args.samples, len(t.df)), random_state=args.seed)
        for row in sample.drop(columns=["source_file"]).to_dict("records"):
            print(f"  row:    {row}")
            print(f"  source: {source_line_for(name, row, source_text)}")

    failed = [c for c in results if not c.passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
