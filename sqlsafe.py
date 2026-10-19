"""Read-only SQL execution against nba_stats.db for the agent's run_sql tool.

Layers, each enough on its own to stop a write:
1. Text checks: one SELECT/WITH statement, no comments or semicolons, no write/admin keywords.
2. The database is opened with a read-only URI (mode=ro) and PRAGMA query_only=ON.
3. A SQLite authorizer that only permits reads and function calls.
Queries are capped at MAX_ROWS rows and interrupted after TIMEOUT_SECONDS via a progress handler.
"""

from __future__ import annotations

import re
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_ROWS: int = 50
TIMEOUT_SECONDS: float = 5.0
PROGRESS_EVERY_N_OPS: int = 1_000

FORBIDDEN_KEYWORDS: frozenset[str] = frozenset({
    "ATTACH", "DETACH", "PRAGMA", "INSERT", "UPDATE", "DELETE", "DROP",
    "ALTER", "CREATE", "REPLACE", "VACUUM", "REINDEX", "ANALYZE", "UPSERT",
})

_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'")
_WORD_RE = re.compile(r"[A-Za-z_]+")

_ALLOWED_ACTIONS: frozenset[int] = frozenset({
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
})


class UnsafeSQLError(ValueError):
    """The SQL text is not a single read-only SELECT/WITH statement."""


class QueryTimeoutError(RuntimeError):
    """The query ran longer than the allowed time and was interrupted."""


@dataclass(frozen=True)
class QueryResult:
    """Rows returned by a query, capped at max_rows."""

    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool


def validate_sql(sql: str) -> str:
    """Checks that `sql` is one read-only SELECT/WITH statement and returns it stripped.

    One trailing ";" (LLMs usually add it) is removed first; any other semicolon is rejected.
    String literals are blanked before scanning, so a value like 'Drop' or 'a;b' is allowed.

    Raises:
        UnsafeSQLError: With a message the LLM can act on.
    """
    stripped = sql.strip()
    if stripped.endswith(";"):
        stripped = stripped[:-1].rstrip()
    if not stripped:
        raise UnsafeSQLError("Empty query.")

    if stripped.count("'") % 2 != 0:
        raise UnsafeSQLError("Unterminated string literal.")
    code = _STRING_LITERAL_RE.sub("''", stripped)

    if "--" in code or "/*" in code or "*/" in code:
        raise UnsafeSQLError("SQL comments are not allowed.")
    if ";" in code:
        raise UnsafeSQLError("Semicolons are only allowed at the very end: send exactly one statement.")

    words = [w.upper() for w in _WORD_RE.findall(code)]
    if not words or words[0] not in {"SELECT", "WITH"}:
        raise UnsafeSQLError("Only a single SELECT (or WITH ... SELECT) query is allowed.")
    forbidden = sorted(FORBIDDEN_KEYWORDS.intersection(words))
    if forbidden:
        raise UnsafeSQLError(f"Forbidden keyword(s): {', '.join(forbidden)}. Only read-only SELECT queries are allowed.")

    return stripped


def _authorizer(action: int, *_args: Any) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def open_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Opens an existing SQLite file read-only (safe for paths with spaces) with query_only enabled.

    Raises:
        FileNotFoundError: If the database file does not exist (it is never created).
    """
    db_file = Path(db_path).resolve()
    if not db_file.is_file():
        raise FileNotFoundError(f"Database '{db_file.name}' not found. Load it with: python extract_tables.py load")
    conn = sqlite3.connect(db_file.as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def run_query(
    sql: str,
    db_path: str | Path,
    max_rows: int = MAX_ROWS,
    timeout_s: float = TIMEOUT_SECONDS,
) -> QueryResult:
    """Validates and runs one read-only query, returning at most `max_rows` rows.

    Raises:
        UnsafeSQLError: If the query fails validation or touches anything but reads.
        QueryTimeoutError: If the query runs longer than `timeout_s`.
        FileNotFoundError: If the database does not exist.
        sqlite3.Error: For ordinary SQL errors (unknown column, syntax error, ...).
    """
    query = validate_sql(sql)
    with closing(open_readonly(db_path)) as conn:
        conn.set_authorizer(_authorizer)
        deadline = time.monotonic() + timeout_s
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), PROGRESS_EVERY_N_OPS)
        try:
            cursor = conn.execute(query)
            fetched = cursor.fetchmany(max_rows + 1)
        except sqlite3.OperationalError as e:
            if time.monotonic() > deadline and "interrupt" in str(e).lower():
                raise QueryTimeoutError(
                    f"Query took longer than {timeout_s:g}s and was stopped. Simplify it or add a LIMIT."
                ) from e
            raise
        except sqlite3.DatabaseError as e:
            if "not authorized" in str(e).lower():
                raise UnsafeSQLError("Query tried a non-read operation, which is not allowed.") from e
            raise
        columns = [d[0] for d in cursor.description or []]

    return QueryResult(columns=columns, rows=fetched[:max_rows], truncated=len(fetched) > max_rows)


def _cell(value: Any) -> str:
    if value is None:
        return "NULL"
    return str(value).replace("|", "\\|").replace("\n", " ")


def to_markdown(result: QueryResult) -> str:
    """Renders a QueryResult as a markdown table, noting when rows were cut off."""
    if not result.columns:
        return "(query returned no columns)"
    if not result.rows:
        return "(0 rows)"

    lines = [
        "| " + " | ".join(_cell(c) for c in result.columns) + " |",
        "| " + " | ".join("---" for _ in result.columns) + " |",
    ]
    lines.extend("| " + " | ".join(_cell(v) for v in row) + " |" for row in result.rows)
    if result.truncated:
        lines.append(f"\n(Showing the first {len(result.rows)} rows only. Use LIMIT, filters or aggregates.)")
    return "\n".join(lines)


def safe_query(sql: str, db_path: str | Path, max_rows: int = MAX_ROWS, timeout_s: float = TIMEOUT_SECONDS) -> str:
    """Runs a validated read-only query and returns the rows as a markdown table."""
    return to_markdown(run_query(sql, db_path, max_rows=max_rows, timeout_s=timeout_s))


# --------------------------------------------------------------------------------------------------
# Source-page footer: which PDF pages the tables in a query come from
# --------------------------------------------------------------------------------------------------

SOURCE_PAGE_COLUMN = "source_page"


def referenced_tables(sql: str, table_names: list[str]) -> list[str]:
    """Known tables whose names appear in `sql` as identifiers, outside single-quoted string literals.

    Matches whole words only, case-insensitively, bare or "double-quoted", in order of first appearance.
    """
    code = _STRING_LITERAL_RE.sub("''", sql)
    found: list[tuple[int, str]] = []
    for name in table_names:
        match = re.search(rf"(?<![\w$]){re.escape(name)}(?![\w$])", code, re.IGNORECASE)
        if match:
            found.append((match.start(), name))
    return [name for _, name in sorted(found)]


def format_pages(pages: list[int]) -> str:
    """[179, 180] -> "PDF pp. 179-180"; [52] -> "PDF p. 52"; [6, 9, 10] -> "PDF pp. 6, 9-10"."""
    ranges: list[str] = []
    ordered = sorted(set(pages))
    start = prev = ordered[0]
    for page in [*ordered[1:], None]:
        if page is not None and page == prev + 1:
            prev = page
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        if page is not None:
            start = prev = page
    return f"PDF {'p.' if len(ordered) == 1 else 'pp.'} {', '.join(ranges)}"


def source_footer(sql: str, db_path: str | Path, timeout_s: float = TIMEOUT_SECONDS) -> str:
    """'Sources: table = PDF pp. ...' for each known table the query references that has a source_page column.

    Every lookup goes through run_query (validated, read-only, authorizer, timeout, row cap).
    Returns "" when no referenced table has source pages.
    """
    tables = [str(r[0]) for r in run_query(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
        db_path, timeout_s=timeout_s,
    ).rows]
    parts: list[str] = []
    for table in referenced_tables(sql, tables):
        ident = '"' + table.replace('"', '""') + '"'
        # PRAGMA table_info is denied by the authorizer, so a table without the column is detected by its error.
        try:
            pages = [int(r[0]) for r in run_query(
                f"SELECT DISTINCT {SOURCE_PAGE_COLUMN} FROM {ident} WHERE {SOURCE_PAGE_COLUMN} IS NOT NULL "
                f"ORDER BY {SOURCE_PAGE_COLUMN}", db_path, timeout_s=timeout_s,
            ).rows]
        except sqlite3.OperationalError as e:
            if "no such column" not in str(e):
                raise
            continue
        if pages:
            parts.append(f"{table} = {format_pages(pages)}")
    return f"Sources: {'; '.join(parts)}" if parts else ""
