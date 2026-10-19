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
