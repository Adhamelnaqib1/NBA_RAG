"""Table extraction and discovery module for NBA Guide PDF.

Extracts tabular data from PDF pages using pdfplumber, outputs markdown
reports and CSV previews for visual inspection, and loads structured tables
into SQLite with validation and cleaning rules.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sqlite3
import sys
import unicodedata
from contextlib import closing
from dataclasses import dataclass, field
from itertools import pairwise
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd  # type: ignore[import-untyped]
import pdfplumber
import yaml
from pdfplumber.utils import cluster_objects

if TYPE_CHECKING:
    from validate_tables import Check

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Table Cleaning Helpers
# ----------------------------------------------------------------------


def clean_number(val: Any) -> int | float | None:
    """Converts a raw string representation of a number to int or float.

    Handles formatted strings with commas ("1,234"), percentages, decimals,
    and negative values. Returns None for empty or invalid strings.

    Args:
        val: Input value of any type.

    Returns:
        Converted int, float, or None if conversion is not possible.
    """
    if val is None or pd.isna(val):
        return None
    s = str(val).strip().replace(",", "").rstrip(".")
    if not s or s in {"-", "–", "N/A", "nan", "None"}:
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return None


def split_w_l(val: str) -> tuple[int | None, int | None]:
    """Splits a Win-Loss string into integer win and loss counts.

    Example inputs: "60-22", "4-1", "24- 17", "11-30".

    Args:
        val: Raw string containing two numbers separated by a hyphen.

    Returns:
        Tuple of (wins, losses) as integers or (None, None) if parsing fails.
    """
    if not val or not isinstance(val, str):
        return None, None
    clean = val.replace(" ", "").replace("–", "-")
    match = re.search(r"(\d+)-(\d+)", clean)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def split_team_coach(val: str) -> tuple[str, str]:
    """Splits a 'Team (Coach)' string into separate team and coach strings.

    Example: 'Philadelphia (Eddie Gottlieb)' -> ('Philadelphia', 'Eddie Gottlieb')
             '*Boston (Doc Rivers)' -> ('Boston', 'Doc Rivers')

    Args:
        val: Raw team and coach string.

    Returns:
        Tuple of (team, coach).
    """
    if not val or not isinstance(val, str):
        return "", ""
    s = val.strip().lstrip("*")
    match = re.match(r"^(.*?)\s*\(([^)]+)\)$", s)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return s, ""


def drop_repeated_headers(df: pd.DataFrame, header_names: list[str]) -> pd.DataFrame:
    """Drops repeated header rows appearing across page breaks.

    Args:
        df: Input DataFrame.
        header_names: List of column names that characterize header rows.

    Returns:
        DataFrame without repeated header rows.
    """
    if df.empty or not header_names:
        return df

    mask = pd.Series(True, index=df.index)
    first_col = header_names[0]
    if first_col in df.columns:
        mask = ~df[first_col].astype(str).str.strip().str.lower().isin([h.lower() for h in header_names])
    return df[mask].reset_index(drop=True)


# ----------------------------------------------------------------------
# Table Discovery
# ----------------------------------------------------------------------


def guess_heading_from_page(page: pdfplumber.page.Page, table_rows: list[list[str | None]]) -> str:
    """Guesses the subject or heading for a table based on surrounding text and rows.

    Args:
        page: The pdfplumber Page object.
        table_rows: The raw extracted rows of the table.

    Returns:
        A concise string representing the best guess at the table's contents.
    """
    page_text = page.extract_text() or ""
    lines = [line.strip() for line in page_text.split("\n") if line.strip()]

    header_lines = [line for line in lines[:5] if not (line.startswith("–") or line.isdigit())]
    heading = " - ".join(header_lines[:2]) if header_lines else "Unknown Table"

    if table_rows and len(table_rows) > 0:
        non_empty = [c.strip() for c in table_rows[0] if c and c.strip()]
        if len(non_empty) == 1 and len(non_empty[0]) > 4:
            heading = f"{heading} ({non_empty[0]})"

    return heading


def extract_page_tables(page: pdfplumber.page.Page) -> list[list[list[str | None]]]:
    """Extracts tables from a page using line and text strategies.

    Args:
        page: The pdfplumber Page object.

    Returns:
        List of tables, where each table is a list of rows containing cell strings.
    """
    tables = page.extract_tables()
    valid_tables = [
        t for t in tables
        if len(t) >= 2 and any(len([c for c in row if c and str(c).strip()]) >= 2 for row in t)
    ]

    if not valid_tables:
        text_tables = page.extract_tables(
            table_settings={
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_y_tolerance": 3,
                "intersection_y_tolerance": 3,
            }
        )
        valid_tables = [
            t for t in text_tables
            if len(t) >= 2 and any(len([c for c in row if c and str(c).strip()]) >= 2 for row in t)
        ]

    return valid_tables


def discover_tables(
    pdf_path: str | Path,
    start_page: int | None = None,
    end_page: int | None = None,
    output_report: str | Path = "table_report.md",
    preview_dir: str | Path = "table_preview",
) -> list[dict[str, Any]]:
    """Discovers tables within a PDF and outputs a report and CSV previews.

    Args:
        pdf_path: Path to the target PDF file.
        start_page: 1-indexed starting page number (inclusive).
        end_page: 1-indexed ending page number (inclusive).
        output_report: Filepath for the generated Markdown report.
        preview_dir: Directory where table CSV previews will be stored.

    Returns:
        List of discovered table metadata dictionaries.

    Raises:
        FileNotFoundError: If the specified PDF does not exist.
        ValueError: If page range arguments are invalid.
    """
    pdf_file = Path(pdf_path).resolve()
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")

    preview_path = Path(preview_dir).resolve()
    preview_path.mkdir(parents=True, exist_ok=True)

    report_path = Path(output_report).resolve()
    discovered_tables: list[dict[str, Any]] = []

    with pdfplumber.open(pdf_file) as pdf:
        total_pdf_pages = len(pdf.pages)
        first_page = max(1, start_page or 1)
        last_page = min(total_pdf_pages, end_page or total_pdf_pages)

        if first_page > last_page:
            raise ValueError(f"start_page ({first_page}) cannot exceed end_page ({last_page}).")

        logger.info(
            "Scanning '%s' for tables across pages %d to %d (total PDF pages: %d)...",
            pdf_file.name,
            first_page,
            last_page,
            total_pdf_pages,
        )

        for page_num in range(first_page, last_page + 1):
            page_idx = page_num - 1
            page = pdf.pages[page_idx]

            try:
                tables = extract_page_tables(page)
            except (ValueError, TypeError, IndexError) as e:
                logger.warning("Error extracting tables from page %d: %s", page_num, e)
                continue

            if not tables:
                continue

            for table_idx, table_data in enumerate(tables, start=1):
                row_count = len(table_data)
                col_count = max(len(row) for row in table_data) if table_data else 0

                first_row = table_data[0] if table_data else []
                header_preview = [
                    str(c).strip().replace("\n", " ")
                    for c in first_row
                    if c is not None and str(c).strip()
                ]
                header_preview_str = " | ".join(header_preview[:8])
                if len(header_preview) > 8:
                    header_preview_str += " ..."

                guess = guess_heading_from_page(page, table_data)
                csv_filename = f"{page_num}_{table_idx}.csv"
                csv_file_path = preview_path / csv_filename

                with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for row in table_data:
                        writer.writerow([cell if cell is not None else "" for cell in row])

                discovered_tables.append(
                    {
                        "page": page_num,
                        "table_num": table_idx,
                        "row_count": row_count,
                        "col_count": col_count,
                        "guess": guess,
                        "header_preview": header_preview_str,
                        "csv_filename": csv_filename,
                        "csv_path": str(csv_file_path),
                    }
                )

    _write_report(
        report_path=report_path,
        pdf_name=pdf_file.name,
        first_page=first_page,
        last_page=last_page,
        discovered_tables=discovered_tables,
    )

    logger.info(
        "Discovery complete. Found %d tables. Report written to: %s",
        len(discovered_tables),
        report_path,
    )
    return discovered_tables


def _write_report(
    report_path: Path,
    pdf_name: str,
    first_page: int,
    last_page: int,
    discovered_tables: list[dict[str, Any]],
) -> None:
    """Formats and writes the Markdown discovery report."""
    lines: list[str] = [
        "# Discovered Tables Report",
        "",
        f"- **Source PDF**: `{pdf_name}`",
        f"- **Page Range**: Pages {first_page} to {last_page} ({last_page - first_page + 1} pages scanned)",
        f"- **Total Tables Discovered**: {len(discovered_tables)}",
        "- **CSV Previews Directory**: `table_preview/`",
        "",
        "## Summary of Tables",
        "",
        "| Page | Table # | Rows | Cols | Guessed Content / Heading | First-Row Preview | Preview Link |",
        "|---|---|---|---|---|---|---|",
    ]

    for item in discovered_tables:
        page = item["page"]
        t_num = item["table_num"]
        rows = item["row_count"]
        cols = item["col_count"]
        guess = item["guess"].replace("|", "/")
        preview = item["header_preview"].replace("|", "/")
        csv_file = item["csv_filename"]
        link = f"[{csv_file}](file://{item['csv_path']})"
        lines.append(f"| {page} | {t_num} | {rows} | {cols} | {guess} | {preview} | {link} |")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------
# Table Loading and Extraction Strategies
# ----------------------------------------------------------------------


LINE_TOLERANCE: float = 3.0
"""Max vertical distance (pt) between word tops that still counts as the same printed line."""


@dataclass(frozen=True)
class SkippedLine:
    """A printed line that produced no table row, kept so validation can review it."""

    page: int
    text: str
    reason: str


@dataclass
class ExtractedTable:
    """A cleaned table plus every source line that was skipped while extracting it."""

    df: pd.DataFrame
    skipped: list[SkippedLine] = field(default_factory=list)


def _skip(skipped: list[SkippedLine], page: int, text: str, reason: str) -> None:
    logger.info("p.%d skipped line (%s): %r", page, reason, text)
    skipped.append(SkippedLine(page=page, text=text, reason=reason))


def _crop(page: pdfplumber.page.Page, bbox: list[float]) -> pdfplumber.page.Page:
    """Crops a page to a [x0, top, x1, bottom] bbox read from the YAML spec."""
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 numbers [x0, top, x1, bottom], got {bbox!r}")
    x0, top, x1, bottom = bbox
    return page.crop((x0, top, x1, bottom))


def cluster_lines(
    words: list[dict[str, Any]],
    tolerance: float = LINE_TOLERANCE,
) -> list[list[dict[str, Any]]]:
    """Groups words into printed lines by their top coordinate, top to bottom, each sorted left to right.

    Unlike bucketing with round(top / n) * n, words whose tops differ by less than
    `tolerance` always land on the same line, even across a bucket boundary.
    """
    clusters = cluster_objects(words, itemgetter("top"), tolerance)
    clusters.sort(key=lambda line: min(w["top"] for w in line))
    return [sorted(line, key=itemgetter("x0")) for line in clusters]


def _line_text(line: list[dict[str, Any]]) -> str:
    return " ".join(w["text"] for w in line)


def _extract_word_columns(
    page: pdfplumber.page.Page,
    page_num: int,
    column_splits: dict[str, list[int]],
    bbox: list[float] | None = None,
) -> tuple[list[dict[str, Any]], list[SkippedLine]]:
    """Extracts rows that start with a pick number, assigning words to columns by x position."""
    target_page = _crop(page, bbox) if bbox else page
    rows: list[dict[str, Any]] = []
    skipped: list[SkippedLine] = []

    for line in cluster_lines(target_page.extract_words()):
        text = _line_text(line)
        first_token = line[0]["text"].rstrip(".")
        if not first_token.isdigit():
            _skip(skipped, page_num, text, "no leading pick number")
            continue

        unassigned = [
            w["text"] for w in line if not any(lo <= w["x0"] < hi for lo, hi in column_splits.values())
        ]
        if unassigned:
            _skip(skipped, page_num, text, f"words outside every column: {unassigned}")
            continue

        row: dict[str, Any] = {
            col: " ".join(w["text"] for w in line if lo <= w["x0"] < hi).strip()
            for col, (lo, hi) in column_splits.items()
        }
        row["pick"] = int(first_token)
        rows.append(row)

    return rows, skipped


def _extract_stacked_blocks(
    page: pdfplumber.page.Page,
    page_num: int,
    blocks: list[dict[str, Any]],
    y_range: list[float],
) -> tuple[list[dict[str, Any]], list[SkippedLine]]:
    """Extracts side-by-side "Player Total 1st 2nd 3rd" blocks, stacking them top to bottom, left to right."""
    words = page.extract_words()
    y_min, y_max = y_range
    rows: list[dict[str, Any]] = []
    skipped: list[SkippedLine] = []

    for block in blocks:
        x_min, x_max = block["x_range"]
        block_words = [w for w in words if x_min <= w["x0"] < x_max and y_min < w["top"] < y_max]

        for line in cluster_lines(block_words):
            tokens = [w["text"] for w in line]
            text = " ".join(tokens)
            numbers, name = tokens[-4:], tokens[:-4]
            if not name or not all(t.isdigit() for t in numbers):
                _skip(skipped, page_num, text, "not '<player> <total> <1st> <2nd> <3rd>'")
                continue
            total, first, second, third = (int(t) for t in numbers)
            rows.append({
                "player": " ".join(name),
                "total": total,
                "first_team": first,
                "second_team": second,
                "third_team": third,
            })

    return rows, skipped


def _extract_line_regex(
    page: pdfplumber.page.Page,
    page_num: int,
    pattern: str,
    fields: list[str],
    bbox: list[float] | None = None,
) -> tuple[list[dict[str, Any]], list[SkippedLine]]:
    """Extracts rows from the page's text lines using a regular expression pattern."""
    target_page = _crop(page, bbox) if bbox else page
    text = target_page.extract_text(y_tolerance=LINE_TOLERANCE) or ""
    regex = re.compile(pattern)
    rows: list[dict[str, Any]] = []
    skipped: list[SkippedLine] = []

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        match = regex.match(line)
        if match:
            rows.append(dict(zip(fields, match.groups())))
        else:
            _skip(skipped, page_num, line, "no regex match")

    return rows, skipped


SEASON_RE = re.compile(r"^(\d{4})-(\d{2})$")
TEAM_LEVELS: dict[str, str] = {"FIRST": "First", "SECOND": "Second", "THIRD": "Third"}
# "Player, Team" with an optional " (tie)" marker; teams may contain spaces, dots, slashes and hyphens.
ENTRY_RE = re.compile(r"^(?P<player>[^,]+), (?P<team>[A-Z][A-Za-z.'’/ -]*?)(?: \((?P<tie>tie)\))?$")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})


def season_end_year(season: str) -> int:
    """Ending year of a season written like "1996-97" (1997); handles the rollover ("1999-00" -> 2000).

    Raises:
        ValueError: If `season` is not "YYYY-YY" or the two years are not consecutive.
    """
    match = SEASON_RE.match(season)
    if not match:
        raise ValueError(f"Not a season like '1996-97': {season!r}")
    start, end_2 = int(match.group(1)), int(match.group(2))
    end = start // 100 * 100 + end_2
    if end <= start:
        end += 100
    if end != start + 1:
        raise ValueError(f"Season years are not consecutive: {season!r}")
    return end


def normalize_player_name(name: str) -> str:
    """Search key for player names: lowercase ASCII, straight apostrophes, no dots/commas, single spaces.

    "Shaquille O’Neal" and "Shaquille O'Neal" -> "shaquille o'neal"; "Karim López" -> "karim lopez";
    "Mikel Brown Jr." and "Mikel Brown, Jr" -> "mikel brown jr".
    """
    folded = unicodedata.normalize("NFKD", name.translate(_APOSTROPHES))
    ascii_only = "".join(c for c in folded if not unicodedata.combining(c)).encode("ascii", "ignore").decode()
    cleaned = re.sub(r"[^a-z0-9' -]", " ", ascii_only.lower().replace(".", " ").replace(",", " "))
    return re.sub(r"\s+", " ", cleaned).strip()


def find_gutters(words: list[dict[str, Any]], min_gap: float) -> list[float]:
    """Midpoints of vertical strips at least `min_gap` wide that no word overlaps, left to right."""
    if not words:
        return []
    spans = sorted((float(w["x0"]), float(w["x1"])) for w in words)
    gutters: list[float] = []
    covered_to = spans[0][1]
    for x0, x1 in spans[1:]:
        if x0 - covered_to >= min_gap:
            gutters.append((covered_to + x0) / 2)
        covered_to = max(covered_to, x1)
    return gutters


def _extract_season_blocks(
    page: pdfplumber.page.Page,
    page_num: int,
    columns: int,
    y_range: list[float],
    min_gutter: float,
) -> tuple[list[dict[str, Any]], list[SkippedLine]]:
    """Extracts season blocks laid out in `columns` columns, read down each column, left to right.

    Each block is a "YYYY-YY" header, then FIRST/SECOND/THIRD labels, each followed by "Player, Team"
    lines. Entries are aligned on the comma, so first names hang left of the header: columns are split
    at the page's empty vertical gutters instead of fixed x positions.

    Raises:
        ValueError: If the page does not have exactly `columns - 1` gutters.
    """
    y_min, y_max = y_range
    words = [w for w in page.extract_words() if y_min < w["top"] < y_max]
    gutters = find_gutters(words, min_gutter)
    if len(gutters) != columns - 1:
        raise ValueError(f"PDF p. {page_num}: expected {columns - 1} column gutters, found {len(gutters)}")
    bounds = [float("-inf"), *gutters, float("inf")]

    rows: list[dict[str, Any]] = []
    skipped: list[SkippedLine] = []
    for lo, hi in pairwise(bounds):
        season: str | None = None
        level: str | None = None
        for line in cluster_lines([w for w in words if lo <= w["x0"] < hi]):
            text = _line_text(line)
            if SEASON_RE.match(text):
                season, level = text, None
            elif text in TEAM_LEVELS:
                level = TEAM_LEVELS[text]
                if season is None:
                    _skip(skipped, page_num, text, "team label before any season header")
            elif season is None or level is None:
                _skip(skipped, page_num, text, "outside a season block")
            elif match := ENTRY_RE.match(text):
                rows.append({
                    "season": season,
                    "season_end_year": season_end_year(season),
                    "team_level": level,
                    "player": match.group("player").strip(),
                    "team": match.group("team").strip(),
                    "is_tie": int(match.group("tie") is not None),
                })
            else:
                _skip(skipped, page_num, text, f"not 'Player, Team' in {season} {level}")
    return rows, skipped


AWARD_ENTRY_RE = re.compile(r"^(?P<label>\d{4}(?:-\d{2})?)\s+—\s+(?P<rest>.+)$")
# "Player, Team (tie)" or "Player (Team)" (the second form is used for co-recipients, e.g. Lanier 2019-20).
AWARD_WINNER_RE = re.compile(
    r"^(?P<player>[^,()]+?)(?:,\s+(?P<team>[A-Z][A-Za-z.'’/ -]*?)|\s+\((?P<pteam>[A-Z][A-Za-z.'’/ -]*)\))"
    r"(?:\s+\((?P<tie>tie)\))?$"
)
_SMALL_WORDS = frozenset({"of", "the", "and", "for", "a", "an", "in", "on"})


def normalize_heading(text: str) -> str:
    """Joins letters split by the PDF text layer ("MOST V ALUABLE" -> "MOST VALUABLE") and collapses spaces."""
    joined = re.sub(r"\b([A-Z]) (?=[A-Z]{2,}\b)", r"\1", text)
    return " ".join(joined.split())


def title_case_heading(text: str) -> str:
    """"NBA FINALS MOST VALUABLE PLAYER" -> "NBA Finals Most Valuable Player"; keeps NBA, initials, hyphen parts."""
    words = []
    for i, word in enumerate(normalize_heading(text).split()):
        if word in ("NBA", "MVP") or re.fullmatch(r"[A-Z]\.", word):
            words.append(word)
        elif i > 0 and word.lower() in _SMALL_WORDS:
            words.append(word.lower())
        else:
            words.append("-".join(part.capitalize() for part in word.split("-")))
    return " ".join(words)


def award_end_year(label: str) -> int:
    """"2025-26" -> 2026 (season awards); "2026" -> 2026 (calendar-year awards such as Finals MVP)."""
    return season_end_year(label) if "-" in label else int(label)


def _extract_award_blocks(
    page: pdfplumber.page.Page,
    page_num: int,
    y_range: list[float],
    min_gutter: float,
    ignore_headings: list[str],
    drop_line_patterns: list[str],
) -> tuple[list[dict[str, Any]], list[SkippedLine]]:
    """Award lists: each bold heading owns the band below it (down to the next heading); the band's entries are
    read column by column, split at the band's own gutters, so a heading always stays with its list.

    Entries are "YYYY-YY — Player, Team" (season awards) or "YYYY — Player, Team" (calendar-year awards); a line
    without a year continues the entry above (a tie or co-recipient). "N/A (...)" years are skipped as not awarded.
    """
    y_min, y_max = y_range
    words = [w for w in page.extract_words(extra_attrs=["fontname", "size"]) if y_min < w["top"] < y_max]
    ignore = {normalize_heading(h) for h in ignore_headings}
    drops = [re.compile(p) for p in drop_line_patterns]
    lines = cluster_lines(words)
    headings = [
        line for line in lines
        if all("Bold" in w["fontname"] for w in line) and min(w["size"] for w in line) >= 9.5
        and normalize_heading(_line_text(line)) not in ignore
    ]
    rows: list[dict[str, Any]] = []
    skipped: list[SkippedLine] = []
    tops = [h[0]["top"] for h in headings] + [y_max]
    heading_ids = {id(w) for h in headings for w in h}
    for heading, (lo, hi) in zip(headings, pairwise(tops)):
        award = title_case_heading(_line_text(heading))
        band_lines = [
            line for line in cluster_lines([w for w in words if lo < w["top"] < hi and id(w) not in heading_ids])
            if not any(d.search(_line_text(line)) for d in drops)
        ]
        band = [w for line in band_lines for w in line]
        bounds = [float("-inf"), *find_gutters(band, min_gutter), float("inf")]
        for col_lo, col_hi in pairwise(bounds):
            label: str | None = None
            for line in cluster_lines([w for w in band if col_lo <= w["x0"] < col_hi]):
                text = _line_text(line)
                entry = AWARD_ENTRY_RE.match(text)
                if entry:
                    label, rest = entry.group("label"), entry.group("rest")
                elif label is not None:
                    rest = text
                else:
                    _skip(skipped, page_num, text, f"no year before this line in {award}")
                    continue
                if rest.startswith("N/A"):
                    _skip(skipped, page_num, text, f"not awarded in {label}: {rest}")
                    continue
                winner = AWARD_WINNER_RE.match(rest)
                if not winner:
                    _skip(skipped, page_num, text, f"not 'Player, Team' in {award} {label}")
                    continue
                rows.append({
                    "award": award,
                    "season_label": label,
                    "season_end_year": award_end_year(label),
                    "player": winner.group("player").strip(),
                    "team": (winner.group("team") or winner.group("pteam") or "").strip(),
                    "is_tie": int(winner.group("tie") is not None),
                })
    return rows, skipped


def load_and_clean_table(
    pdf_path: str | Path,
    table_name: str,
    table_spec: dict[str, Any],
) -> ExtractedTable:
    """Extracts, cleans, and structures a table according to its YAML spec.

    Args:
        pdf_path: Path to source PDF.
        table_name: Target SQLite table identifier.
        table_spec: Configuration dictionary from tables.yaml.

    Returns:
        Cleaned DataFrame (with source_page and source_file added) and all skipped source lines.
    """
    pdf_file = Path(pdf_path).resolve()
    strategy = table_spec.get("strategy")
    all_records: list[dict[str, Any]] = []
    all_skipped: list[SkippedLine] = []

    with pdfplumber.open(pdf_file) as pdf:
        for page_entry in table_spec.get("pages", []):
            page_num = page_entry["page"]
            page = pdf.pages[page_num - 1]
            bbox = page_entry.get("bbox")

            if strategy == "word_columns":
                col_splits = table_spec["column_splits"]
                extracted, skipped = _extract_word_columns(page, page_num, col_splits, bbox=bbox)
            elif strategy == "stacked_column_blocks":
                blocks = page_entry.get("blocks", [])
                y_range = page_entry.get("y_range", [0, float(page.height)])
                extracted, skipped = _extract_stacked_blocks(page, page_num, blocks, y_range)
            elif strategy == "season_blocks":
                extracted, skipped = _extract_season_blocks(
                    page,
                    page_num,
                    columns=int(table_spec["columns_per_page"]),
                    y_range=page_entry.get("y_range", [0, float(page.height)]),
                    min_gutter=float(table_spec.get("min_gutter", 15)),
                )
            elif strategy == "award_blocks":
                extracted, skipped = _extract_award_blocks(
                    page,
                    page_num,
                    y_range=page_entry.get("y_range", [0, float(page.height)]),
                    min_gutter=float(table_spec.get("min_gutter", 12)),
                    ignore_headings=list(table_spec.get("ignore_headings", [])),
                    drop_line_patterns=list(table_spec.get("drop_line_patterns", [])),
                )
            elif strategy == "line_regex":
                pattern = table_spec["line_pattern"]
                fields = table_spec["line_fields"]
                extracted, skipped = _extract_line_regex(page, page_num, pattern, fields, bbox=bbox)
            else:
                raise ValueError(f"Unknown extraction strategy '{strategy}' for table '{table_name}'")

            for rec in extracted:
                rec["source_page"] = page_num
                rec["source_file"] = pdf_file.name

            all_records.extend(extracted)
            all_skipped.extend(skipped)

    df = pd.DataFrame(all_records)
    if df.empty:
        return ExtractedTable(df, all_skipped)

    # Apply cleanup rules
    cleanup = table_spec.get("cleanup", {})
    if cleanup.get("strip_whitespace"):
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].astype(str).str.strip()

    # A leading "*" is a footnote marker; move it into a 0/1 flag column instead of losing it.
    for col, flag_col in cleanup.get("asterisk_flags", {}).items():
        if col in df.columns:
            marked = df[col].astype(str).str.startswith("*")
            df[flag_col] = marked.astype(int)
            df[col] = df[col].astype(str).str.lstrip("*").str.strip()

    # Matching key for names (accents, apostrophes, Jr./Sr.): e.g. {player: player_search}.
    for col, search_col in cleanup.get("search_columns", {}).items():
        if col in df.columns:
            df[search_col] = df[col].astype(str).map(normalize_player_name)

    if cleanup.get("numeric_columns"):
        for col in cleanup["numeric_columns"]:
            if col in df.columns:
                df[col] = df[col].apply(clean_number)

    return ExtractedTable(df, all_skipped)


def load_tables_from_spec(
    pdf_path: str | Path,
    spec_path: str | Path = "tables.yaml",
) -> dict[str, ExtractedTable]:
    """Loads all tables defined in tables.yaml into clean DataFrames.

    Args:
        pdf_path: Path to source PDF.
        spec_path: Path to YAML specification.

    Returns:
        Dictionary mapping table name to its cleaned DataFrame and skipped lines.
    """
    with open(spec_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    tables_spec = spec.get("tables", {})
    loaded_tables: dict[str, ExtractedTable] = {}

    for table_name, t_spec in tables_spec.items():
        logger.info("Extracting and cleaning table '%s'...", table_name)
        loaded_tables[table_name] = load_and_clean_table(
            pdf_path=pdf_path, table_name=table_name, table_spec=t_spec
        )

    return loaded_tables


def save_tables_to_sqlite(
    dfs: dict[str, pd.DataFrame],
    db_path: str | Path = "./nba_stats.db",
) -> None:
    """Writes cleaned DataFrames to SQLite using replace mode.

    Args:
        dfs: Mapping of table names to DataFrames.
        db_path: Path to target SQLite database.
    """
    db_file = Path(db_path).resolve()
    with closing(sqlite3.connect(db_file)) as conn, conn:
        for name, df in dfs.items():
            df.to_sql(name, conn, if_exists="replace", index=False)
            logger.info("Loaded table '%s' (%d rows) into '%s'", name, len(df), db_file.name)


def load_validated(
    pdf_path: str | Path,
    spec_path: str | Path = "tables.yaml",
    db_path: str | Path = "./nba_stats.db",
) -> list[Check]:
    """Extracts the spec's tables and writes them to SQLite only if every validation check passes.

    Args:
        pdf_path: Path to source PDF.
        spec_path: Path to YAML specification.
        db_path: Path to target SQLite database.

    Returns:
        All validation results. The database is untouched when any of them failed.
    """
    # Imported here because validate_tables imports this module.
    from validate_tables import read_source_text, run_checks

    extracted = load_tables_from_spec(pdf_path=pdf_path, spec_path=spec_path)
    results = run_checks(extracted, read_source_text(pdf_path))
    failed = [c for c in results if not c.passed]
    if failed:
        logger.error("Refusing to write '%s': %d validation check(s) failed.", db_path, len(failed))
        return results

    save_tables_to_sqlite({name: t.df for name, t in extracted.items()}, db_path=db_path)
    return results


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return [row[0] for row in rows]


def table_row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Returns {table name: row count} for every user table."""
    return {
        t: int(conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(t)}").fetchone()[0]) for t in _table_names(conn)
    }


def schema_text(db_path: str | Path = "./nba_stats.db", sample_rows: int = 2) -> str:
    """Describes every table for an LLM prompt: name, row count, column names and types, and sample rows.

    Args:
        db_path: Path to the SQLite database file (opened read-only).
        sample_rows: Number of sample rows to show per table.

    Returns:
        Plain-text schema with one section per table and markdown sample rows.
    """
    db_file = Path(db_path).resolve()
    if not db_file.exists():
        return f"Database '{db_file.name}' does not exist."

    sections: list[str] = []
    with closing(sqlite3.connect(db_file.as_uri() + "?mode=ro", uri=True)) as conn:
        for table, count in table_row_counts(conn).items():
            ident = _quote_ident(table)
            # PRAGMA table_info rows: (cid, name, type, notnull, default, pk)
            columns = ", ".join(f"{c[1]} {c[2] or 'ANY'}" for c in conn.execute(f"PRAGMA table_info({ident})"))
            sample = pd.read_sql_query(f"SELECT * FROM {ident} LIMIT {int(sample_rows)}", conn)
            sections.append(
                f"Table: {table} ({count} rows)\n"
                f"Columns: {columns}\n"
                f"Sample rows:\n{sample.to_markdown(index=False)}"
            )

    return "\n\n".join(sections) if sections else "Database contains no tables."


# ----------------------------------------------------------------------
# CLI Interface
# ----------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for table discovery and loading."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Extract and discover tabular data from PDF files into SQLite."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # discover command
    discover_parser = subparsers.add_parser(
        "discover",
        help="Discover tables within a PDF and output report and CSV previews.",
    )
    discover_parser.add_argument("--pdf", required=True, type=str, help="Path to PDF file")
    discover_parser.add_argument(
        "--start-page", type=int, default=None, help="Start page (1-based, inclusive)"
    )
    discover_parser.add_argument(
        "--end-page", type=int, default=None, help="End page (1-based, inclusive)"
    )
    discover_parser.add_argument(
        "--output-report",
        type=str,
        default="table_report.md",
        help="Output markdown report path (default: table_report.md)",
    )
    discover_parser.add_argument(
        "--preview-dir",
        type=str,
        default="table_preview",
        help="Output directory for CSV previews (default: table_preview)",
    )

    # load command
    load_parser = subparsers.add_parser(
        "load",
        help="Extract and load tables into SQLite based on tables.yaml specification.",
    )
    load_parser.add_argument("--pdf", required=True, type=str, help="Path to PDF file")
    load_parser.add_argument(
        "--spec", type=str, default="tables.yaml", help="Path to tables.yaml specification"
    )
    load_parser.add_argument(
        "--db-path", type=str, default="./nba_stats.db", help="Path to SQLite database"
    )
    load_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview extracted tables without writing to database",
    )

    args = parser.parse_args()

    if args.command == "discover":
        discover_tables(
            pdf_path=args.pdf,
            start_page=args.start_page,
            end_page=args.end_page,
            output_report=args.output_report,
            preview_dir=args.preview_dir,
        )
    elif args.command == "load":
        if args.dry_run:
            # Same checks as a real load, without touching the database.
            from validate_tables import read_source_text, run_checks

            results = run_checks(load_tables_from_spec(args.pdf, args.spec), read_source_text(args.pdf))
        else:
            results = load_validated(pdf_path=args.pdf, spec_path=args.spec, db_path=args.db_path)

        for c in results:
            print(f"{'WARN' if c.warning else 'PASS' if c.passed else 'FAIL'}  {c.table}: {c.name}  [{c.detail}]")
        if not all(c.passed for c in results):
            sys.exit("Validation failed; database was not written.")
        if args.dry_run:
            print("\nDry run: all checks passed, database not written.")
            return

        db_file = Path(args.db_path).resolve()
        with closing(sqlite3.connect(db_file.as_uri() + "?mode=ro", uri=True)) as conn:
            print(f"\nWrote {db_file.name}:")
            for name, count in table_row_counts(conn).items():
                print(f"  {name}: {count} rows")
        print("\n" + schema_text(db_file))

if __name__ == "__main__":
    main()
