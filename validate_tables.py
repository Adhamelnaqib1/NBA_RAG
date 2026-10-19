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

from extract_tables import ExtractedTable, SkippedLine, load_tables_from_spec

logger = logging.getLogger(__name__)

SOURCE_PAGES: dict[str, list[int]] = {
    "draft_2026": [52],
    "all_nba_by_player": [111, 112],
    "nba_finals": [179, 180],
}

# Patterns over raw page text, independent of the extractor's bboxes, x-splits and line clustering.
DRAFT_SOURCE_RE = re.compile(r"^(\d+)\. ")
ALL_NBA_SOURCE_RE = re.compile(r"([A-Z][^\d]*?) (\d+) (\d+) (\d+) (\d+)(?= |$)")
FINALS_SOURCE_RE = re.compile(r"^(\d{4}) ")

# A skipped line that matches one of these looked like data, so skipping it lost a row.
DATA_LIKE: dict[str, re.Pattern[str]] = {
    "draft_2026": re.compile(r"^\d+\.?\s"),
    "all_nba_by_player": re.compile(r"\d+ \d+ \d+ \d+$"),
    "nba_finals": re.compile(r"^\d{4}\b"),
}


@dataclass(frozen=True)
class Check:
    """One named validation result."""

    table: str
    name: str
    passed: bool
    detail: str = ""


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


REQUIRED_COLUMNS: dict[str, list[str]] = {
    "draft_2026": ["pick", "player", "drafting_team", "before_nba", "country", "source_page"],
    "all_nba_by_player": ["player", "total", "first_team", "second_team", "third_team", "source_page"],
    "nba_finals": [
        "year", "winner_team", "loser_team", "winner_games", "loser_games",
        "winner_best_record", "loser_best_record", "source_page",
    ],
}

CHECKS: dict[str, Callable[[ExtractedTable, dict[int, str]], list[Check]]] = {
    "draft_2026": check_draft,
    "all_nba_by_player": check_all_nba,
    "nba_finals": check_finals,
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
    return results


def source_line_for(table: str, row: dict[str, Any], source_text: dict[int, str]) -> str:
    """Finds the raw page-text line a row came from (by pick, player name or year)."""
    lines = _source_lines(source_text, [int(row["source_page"])])
    if table == "draft_2026":
        prefix = f"{row['pick']}. "
        match = [ln for ln in lines if ln.startswith(prefix)]
    elif table == "all_nba_by_player":
        match = [ln for ln in lines if re.search(rf"(^| ){re.escape(row['player'])} \d", ln)]
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
        print(f"{'PASS' if c.passed else 'FAIL'}  {c.table}: {c.name}  [{c.detail}]")

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
