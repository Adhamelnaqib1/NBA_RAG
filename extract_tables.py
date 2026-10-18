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
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pdfplumber
import yaml

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


def _extract_word_columns(
    page: pdfplumber.page.Page,
    column_splits: dict[str, list[int]],
    bbox: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Extracts rows by word coordinates based on horizontal splits."""
    target_page = page.crop(bbox) if bbox else page
    words = target_page.extract_words()

    lines_dict: dict[float, list[dict[str, Any]]] = {}
    for w in words:
        y = round(w["top"] / 3.0) * 3
        lines_dict.setdefault(y, []).append(w)

    rows: list[dict[str, Any]] = []
    for y in sorted(lines_dict.keys()):
        line_words = sorted(lines_dict[y], key=lambda item: item["x0"])
        first_token = line_words[0]["text"].rstrip(".")
        if not first_token.isdigit():
            continue

        row_dict: dict[str, Any] = {}
        for col_name, (x0_bound, x1_bound) in column_splits.items():
            matched = [w["text"] for w in line_words if x0_bound <= w["x0"] < x1_bound]
            text = " ".join(matched).strip()
            row_dict[col_name] = text

        if row_dict.get("pick") and row_dict["pick"].rstrip(".").isdigit():
            row_dict["pick"] = int(row_dict["pick"].rstrip("."))
            rows.append(row_dict)

    return rows


def _extract_stacked_blocks(
    page: pdfplumber.page.Page,
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extracts and stacks side-by-side vertical column blocks."""
    words = page.extract_words()
    all_rows: list[dict[str, Any]] = []

    for block in blocks:
        x_min, x_max = block["x_range"]
        splits = block["splits"]

        block_words = [w for w in words if x_min <= w["x0"] < x_max and 45 < w["top"] < 630]
        lines_dict: dict[float, list[dict[str, Any]]] = {}
        for w in block_words:
            y = round(w["top"] / 3.0) * 3
            lines_dict.setdefault(y, []).append(w)

        for y in sorted(lines_dict.keys()):
            lw = sorted(lines_dict[y], key=lambda item: item["x0"])
            player_words = [w["text"] for w in lw if w["x0"] < splits[0]]
            total_w = [w["text"] for w in lw if splits[0] <= w["x0"] < splits[1]]
            first_w = [w["text"] for w in lw if splits[1] <= w["x0"] < splits[2]]
            second_w = [w["text"] for w in lw if splits[2] <= w["x0"] < splits[3]]
            third_w = [w["text"] for w in lw if w["x0"] >= splits[3]]

            player = " ".join(player_words).strip()
            if not player or player in {"Player", "–"}:
                continue

            try:
                total = int("".join(total_w))
                first = int("".join(first_w)) if first_w else 0
                second = int("".join(second_w)) if second_w else 0
                third = int("".join(third_w)) if third_w else 0
                all_rows.append({
                    "player": player,
                    "total": total,
                    "first_team": first,
                    "second_team": second,
                    "third_team": third,
                })
            except ValueError:
                continue

    return all_rows


def _extract_line_regex(
    page: pdfplumber.page.Page,
    pattern: str,
    fields: list[str],
    bbox: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Extracts rows from page text using a regular expression pattern."""
    target_page = page.crop(bbox) if bbox else page
    text = target_page.extract_text() or ""
    regex = re.compile(pattern)

    rows: list[dict[str, Any]] = []
    for line in text.split("\n"):
        line = line.strip()
        match = regex.match(line)
        if match:
            groups = match.groups()
            row = dict(zip(fields, groups))
            rows.append(row)

    return rows


def load_and_clean_table(
    pdf_path: str | Path,
    table_name: str,
    table_spec: dict[str, Any],
) -> pd.DataFrame:
    """Extracts, cleans, and structures a table according to its YAML spec.

    Args:
        pdf_path: Path to source PDF.
        table_name: Target SQLite table identifier.
        table_spec: Configuration dictionary from tables.yaml.

    Returns:
        Cleaned pandas DataFrame with source_page and source_file added.
    """
    pdf_file = Path(pdf_path).resolve()
    strategy = table_spec.get("strategy")
    all_records: list[dict[str, Any]] = []

    with pdfplumber.open(pdf_file) as pdf:
        for page_entry in table_spec.get("pages", []):
            page_num = page_entry["page"]
            page = pdf.pages[page_num - 1]
            bbox = page_entry.get("bbox")

            if strategy == "word_columns":
                col_splits = table_spec["column_splits"]
                extracted = _extract_word_columns(page, col_splits, bbox=bbox)
            elif strategy == "stacked_column_blocks":
                blocks = page_entry.get("blocks", [])
                extracted = _extract_stacked_blocks(page, blocks)
            elif strategy == "line_regex":
                pattern = table_spec["line_pattern"]
                fields = table_spec["line_fields"]
                extracted = _extract_line_regex(page, pattern, fields, bbox=bbox)
            else:
                raise ValueError(f"Unknown extraction strategy '{strategy}' for table '{table_name}'")

            for rec in extracted:
                rec["source_page"] = page_num
                rec["source_file"] = pdf_file.name

            all_records.extend(extracted)

    df = pd.DataFrame(all_records)
    if df.empty:
        return df

    # Apply cleanup rules
    cleanup = table_spec.get("cleanup", {})
    if cleanup.get("strip_whitespace"):
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()

    if cleanup.get("strip_asterisks"):
        for col in cleanup["strip_asterisks"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lstrip("*").str.strip()

    if cleanup.get("numeric_columns"):
        for col in cleanup["numeric_columns"]:
            if col in df.columns:
                df[col] = df[col].apply(clean_number)

    return df


def load_tables_from_spec(
    pdf_path: str | Path,
    spec_path: str | Path = "tables.yaml",
) -> dict[str, pd.DataFrame]:
    """Loads all tables defined in tables.yaml into clean DataFrames.

    Args:
        pdf_path: Path to source PDF.
        spec_path: Path to YAML specification.

    Returns:
        Dictionary mapping table name to cleaned DataFrame.
    """
    with open(spec_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    tables_spec = spec.get("tables", {})
    loaded_tables: dict[str, pd.DataFrame] = {}

    for table_name, t_spec in tables_spec.items():
        logger.info("Extracting and cleaning table '%s'...", table_name)
        df = load_and_clean_table(pdf_path=pdf_path, table_name=table_name, table_spec=t_spec)
        loaded_tables[table_name] = df

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
    with sqlite3.connect(db_file) as conn:
        for name, df in dfs.items():
            df.to_sql(name, conn, if_exists="replace", index=False)
            logger.info("Loaded table '%s' (%d rows) into '%s'", name, len(df), db_file.name)


def schema(db_path: str | Path = "./nba_stats.db") -> str:
    """Returns database schema, column dtypes, and 2 sample rows per table.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Formatted text representation for use in prompts.
    """
    db_file = Path(db_path).resolve()
    if not db_file.exists():
        return f"Database '{db_file}' does not exist."

    output_lines: list[str] = [f"Database Schema ({db_file.name}):", "=" * 40]

    with sqlite3.connect(f"file:{db_file}?mode=ro", uri=True) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]

        if not tables:
            return "Database contains no tables."

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns_info = cursor.fetchall()
            col_desc = ", ".join(f"{col[1]} {col[2]}" for col in columns_info)
            output_lines.append(f"\nTable: {table}")
            output_lines.append(f"Columns: ({col_desc})")

            # Sample rows
            sample_df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 2;", conn)
            output_lines.append("Sample Rows:")
            output_lines.append(sample_df.to_string(index=False))

    return "\n".join(output_lines)


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
        tables = load_tables_from_spec(pdf_path=args.pdf, spec_path=args.spec)
        for t_name, df in tables.items():
            print(f"\nTable '{t_name}': {len(df)} rows")
            print("Null counts per column:")
            print(df.isnull().sum())
            print("Sample 3 rows:")
            print(df.head(3))
            null_pct = df.isnull().mean()
            suspicious = null_pct[null_pct > 0.2].to_dict()
            if suspicious:
                print(f"WARNING: Suspicious high nulls (>20%): {suspicious}")

        if not args.dry_run:
            save_tables_to_sqlite(tables, db_path=args.db_path)
            print("\nDatabase Schema Summary:")
            print(schema(args.db_path))


if __name__ == "__main__":
    main()
