"""Unit tests for sqlsafe.py (local temporary SQLite files only)."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from pathlib import Path

import pytest

import sqlsafe
from sqlsafe import (
    QueryTimeoutError,
    UnsafeSQLError,
    run_query,
    safe_query,
    validate_sql,
)

INFINITE_CTE = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"


@pytest.fixture
def db(tmp_path: Path) -> Path:
    """A small finals table in a directory whose path contains spaces."""
    path = tmp_path / "NBA RAG dir" / "nba stats.db"
    path.parent.mkdir()
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE nba_finals (year INTEGER, winner_team TEXT, note TEXT)")
        conn.executemany(
            "INSERT INTO nba_finals VALUES (?, ?, ?)",
            [(1947 + i, "Boston" if i % 2 else "LA Lakers", None) for i in range(80)],
        )
    return path


class TestValidateSql:
    @pytest.mark.parametrize(
        "sql",
        ["SELECT 1", "  select year from nba_finals  ", "WITH t AS (SELECT 1 AS a) SELECT a FROM t"],
    )
    def test_accepts_single_select_or_with(self, sql: str) -> None:
        assert validate_sql(sql) == sql.strip()

    @pytest.mark.parametrize("keyword", sorted(sqlsafe.FORBIDDEN_KEYWORDS))
    def test_rejects_every_forbidden_keyword(self, keyword: str) -> None:
        with pytest.raises(UnsafeSQLError, match=f"Forbidden keyword.*{keyword}"):
            validate_sql(f"SELECT year FROM nba_finals WHERE {keyword.lower()} = 1")

    @pytest.mark.parametrize(
        "sql",
        [
            "INSERT INTO nba_finals VALUES (1, 'x', NULL)",
            "UPDATE nba_finals SET year = 1",
            "DELETE FROM nba_finals",
            "DROP TABLE nba_finals",
            "ALTER TABLE nba_finals ADD COLUMN x",
            "CREATE TABLE x (a)",
            "REPLACE INTO nba_finals VALUES (1, 'x', NULL)",
            "ATTACH DATABASE 'other.db' AS other",
            "PRAGMA writable_schema = ON",
            "VACUUM",
            "EXPLAIN SELECT 1",
        ],
    )
    def test_rejects_statements_not_starting_with_select(self, sql: str) -> None:
        with pytest.raises(UnsafeSQLError, match="Only a single SELECT"):
            validate_sql(sql)

    def test_rejects_write_hidden_in_with(self) -> None:
        with pytest.raises(UnsafeSQLError, match="DELETE"):
            validate_sql("WITH t AS (SELECT 1) DELETE FROM nba_finals")

    @pytest.mark.parametrize("sql", ["SELECT 1 -- hi", "SELECT /* hi */ 1", "SELECT 1 */"])
    def test_rejects_comments(self, sql: str) -> None:
        with pytest.raises(UnsafeSQLError, match="comments"):
            validate_sql(sql)

    @pytest.mark.parametrize(
        "sql",
        ["SELECT 1; SELECT 2", "SELECT 1; DROP TABLE nba_finals", "SELECT 1;;", "SELECT 1 ; ;", "SELECT ;1"],
    )
    def test_rejects_semicolons_and_multiple_statements(self, sql: str) -> None:
        with pytest.raises(UnsafeSQLError, match="Semicolons"):
            validate_sql(sql)

    @pytest.mark.parametrize(
        ("sql", "expected"),
        [("SELECT 1;", "SELECT 1"), ("  SELECT 1 ;  \n", "SELECT 1"), ("SELECT 'a;';", "SELECT 'a;'")],
    )
    def test_strips_one_trailing_semicolon(self, sql: str, expected: str) -> None:
        assert validate_sql(sql) == expected

    @pytest.mark.parametrize("sql", [";", "  ;  "])
    def test_lone_semicolon_is_empty(self, sql: str) -> None:
        with pytest.raises(UnsafeSQLError, match="Empty query"):
            validate_sql(sql)

    def test_trailing_semicolon_query_runs(self, db: Path) -> None:
        assert run_query("SELECT count(*) FROM nba_finals;", db).rows == [(80,)]

    @pytest.mark.parametrize("sql", ["", "   ", "SELECT 'unterminated"])
    def test_rejects_empty_and_unterminated(self, sql: str) -> None:
        with pytest.raises(UnsafeSQLError):
            validate_sql(sql)

    def test_keywords_and_semicolons_inside_string_literals_are_allowed(self) -> None:
        sql = "SELECT year FROM nba_finals WHERE winner_team = 'Drop; Delete -- it''s fine'"
        assert validate_sql(sql) == sql


class TestRunQuery:
    def test_path_with_space_works(self, db: Path) -> None:
        assert " " in str(db)
        result = run_query("SELECT count(*) AS n FROM nba_finals", db)
        assert result.columns == ["n"]
        assert result.rows == [(80,)]
        assert not result.truncated

    def test_row_cap_is_50(self, db: Path) -> None:
        result = run_query("SELECT year FROM nba_finals ORDER BY year", db)
        assert len(result.rows) == sqlsafe.MAX_ROWS == 50
        assert result.truncated
        assert result.rows[0] == (1947,)

    def test_infinite_recursive_cte_times_out(self, db: Path) -> None:
        start = time.monotonic()
        with pytest.raises(QueryTimeoutError, match="longer than 0.2s"):
            run_query(INFINITE_CTE, db, timeout_s=0.2)
        assert time.monotonic() - start < 3

    def test_missing_db_raises_and_is_not_created(self, tmp_path: Path) -> None:
        missing = tmp_path / "no such.db"
        with pytest.raises(FileNotFoundError, match="not found"):
            run_query("SELECT 1", missing)
        assert not missing.exists()

    def test_sql_error_propagates(self, db: Path) -> None:
        with pytest.raises(sqlite3.OperationalError, match="no such column"):
            run_query("SELECT nope FROM nba_finals", db)

    def test_authorizer_denies_non_read_even_if_validation_is_bypassed(
        self, db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sqlsafe, "validate_sql", lambda sql: sql)
        with pytest.raises(UnsafeSQLError, match="non-read"):
            run_query("DELETE FROM nba_finals", db)
        assert run_query("SELECT count(*) FROM nba_finals", db).rows == [(80,)]


class TestReadOnlyConnection:
    @pytest.mark.parametrize(
        "sql",
        [
            "INSERT INTO nba_finals VALUES (2027, 'x', NULL)",
            "UPDATE nba_finals SET winner_team = 'x'",
            "DELETE FROM nba_finals",
            "DROP TABLE nba_finals",
            "CREATE TABLE evil (a)",
        ],
    )
    def test_writes_fail_on_readonly_connection(self, db: Path, sql: str) -> None:
        with closing(sqlsafe.open_readonly(db)) as conn:
            assert conn.execute("PRAGMA query_only").fetchone() == (1,)
            with pytest.raises(sqlite3.OperationalError, match="readonly|read-only|query_only"):
                conn.execute(sql)
        with closing(sqlite3.connect(db)) as conn:
            assert conn.execute("SELECT count(*) FROM nba_finals").fetchone() == (80,)

    def test_readonly_even_without_query_only(self, db: Path) -> None:
        with closing(sqlsafe.open_readonly(db)) as conn:
            conn.execute("PRAGMA query_only=OFF")
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                conn.execute("DELETE FROM nba_finals")


class TestMarkdown:
    def test_markdown_table_with_nulls_and_pipes(self, db: Path) -> None:
        text = safe_query("SELECT year, 'a|b' AS label, note FROM nba_finals ORDER BY year LIMIT 2", db)
        assert text.splitlines() == [
            "| year | label | note |",
            "| --- | --- | --- |",
            "| 1947 | a\\|b | NULL |",
            "| 1948 | a\\|b | NULL |",
        ]

    def test_markdown_notes_truncation_and_empty(self, db: Path) -> None:
        assert "first 50 rows only" in safe_query("SELECT year FROM nba_finals", db)
        assert safe_query("SELECT year FROM nba_finals WHERE year > 3000", db) == "(0 rows)"


@pytest.fixture
def paged_db(tmp_path: Path) -> Path:
    """Tables with and without source_page, in a path with spaces."""
    path = tmp_path / "paged dir" / "nba stats.db"
    path.parent.mkdir()
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE nba_finals (year INTEGER, winner_team TEXT, source_page INTEGER)")
        conn.executemany("INSERT INTO nba_finals VALUES (?, ?, ?)",
                         [(1947, "Philadelphia", 179), (2026, "New York", 180), (2025, "Oklahoma City", 180)])
        conn.execute("CREATE TABLE all_nba_by_season (season TEXT, player TEXT, source_page INTEGER)")
        conn.executemany("INSERT INTO all_nba_by_season VALUES (?, ?, ?)",
                         [(f"s{p}", f"P{p}", p) for p in (104, 105, 106, 107, 108, 109)])
        conn.execute("CREATE TABLE draft_2026 (pick INTEGER, player TEXT, source_page INTEGER)")
        conn.execute("INSERT INTO draft_2026 VALUES (1, 'AJ Dybantsa', 52)")
        conn.execute("CREATE TABLE notes (text TEXT)")
        conn.execute("INSERT INTO notes VALUES ('no pages here')")
    return path


class TestSourceFooter:
    def test_single_table(self, paged_db: Path) -> None:
        sql = "SELECT COUNT(*) FROM nba_finals WHERE winner_team = 'Boston'"
        assert sqlsafe.source_footer(sql, paged_db) == "Sources: nba_finals = PDF pp. 179-180"

    def test_join_lists_every_table_in_order(self, paged_db: Path) -> None:
        sql = "SELECT s.player FROM all_nba_by_season s JOIN draft_2026 d ON d.player = s.player"
        assert sqlsafe.source_footer(sql, paged_db) == (
            "Sources: all_nba_by_season = PDF pp. 104-109; draft_2026 = PDF p. 52"
        )

    def test_cte_finds_inner_table(self, paged_db: Path) -> None:
        sql = "WITH picks AS (SELECT * FROM draft_2026) SELECT COUNT(*) FROM picks"
        assert sqlsafe.source_footer(sql, paged_db) == "Sources: draft_2026 = PDF p. 52"

    def test_table_name_inside_string_literal_is_ignored(self, paged_db: Path) -> None:
        sql = "SELECT 'nba_finals' AS label, 'it''s draft_2026' AS other FROM all_nba_by_season LIMIT 1"
        assert sqlsafe.source_footer(sql, paged_db) == "Sources: all_nba_by_season = PDF pp. 104-109"

    @pytest.mark.parametrize("sql", ["SELECT 1", "SELECT * FROM no_such_table", "SELECT text FROM notes",
                                     "SELECT nba_finals_count FROM (SELECT 1 AS nba_finals_count)"])
    def test_no_footer_for_unknown_or_unpaged_tables(self, paged_db: Path, sql: str) -> None:
        assert sqlsafe.source_footer(sql, paged_db) == ""

    def test_quoted_and_case_insensitive_names(self, paged_db: Path) -> None:
        assert sqlsafe.source_footer('SELECT * FROM "NBA_FINALS"', paged_db) == "Sources: nba_finals = PDF pp. 179-180"

    def test_footer_queries_use_the_read_only_path(self, paged_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[str] = []
        original = sqlsafe.run_query

        def spy(sql: str, db_path: str | Path, max_rows: int = sqlsafe.MAX_ROWS,
                timeout_s: float = sqlsafe.TIMEOUT_SECONDS) -> sqlsafe.QueryResult:
            seen.append(sql)
            return original(sql, db_path, max_rows=max_rows, timeout_s=timeout_s)

        monkeypatch.setattr(sqlsafe, "run_query", spy)
        sqlsafe.source_footer("SELECT * FROM nba_finals", paged_db)
        assert len(seen) == 2 and seen[0].startswith("SELECT name FROM sqlite_master")
        assert seen[1].startswith('SELECT DISTINCT source_page FROM "nba_finals"')
        with closing(sqlite3.connect(paged_db)) as conn:
            assert conn.execute("SELECT count(*) FROM nba_finals").fetchone() == (3,)

    def test_referenced_tables(self) -> None:
        names = ["draft_2026", "nba_finals", "all_nba_by_player"]
        assert sqlsafe.referenced_tables("SELECT * FROM nba_finals, draft_2026", names) == ["nba_finals", "draft_2026"]
        assert sqlsafe.referenced_tables("SELECT 'all_nba_by_player' FROM x_nba_finals_y", names) == []

    @pytest.mark.parametrize(
        ("pages", "text"),
        [([52], "PDF p. 52"), ([180, 179], "PDF pp. 179-180"), ([6, 9, 10], "PDF pp. 6, 9-10"),
         ([104, 105, 106, 107, 108, 109], "PDF pp. 104-109"), ([111, 111, 112], "PDF pp. 111-112")],
    )
    def test_format_pages(self, pages: list[int], text: str) -> None:
        assert sqlsafe.format_pages(pages) == text
