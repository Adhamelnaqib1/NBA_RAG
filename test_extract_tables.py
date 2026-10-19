"""Unit tests for extract_tables.py discovery functionality."""

from __future__ import annotations

import csv
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml

import extract_tables
from validate_tables import Check


class TestTableDiscovery:
    """Tests for table extraction, guessing, and report generation."""

    def test_missing_pdf_raises_file_not_found(self, tmp_path: Path) -> None:
        """Verifies FileNotFoundError when PDF does not exist."""
        with pytest.raises(FileNotFoundError):
            extract_tables.discover_tables(
                pdf_path=tmp_path / "nonexistent.pdf",
                preview_dir=tmp_path / "preview",
            )

    def test_invalid_page_range_raises_value_error(self, tmp_path: Path) -> None:
        """Verifies ValueError when start_page > end_page."""
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 dummy")

        mock_doc = MagicMock()
        mock_doc.pages = [MagicMock(), MagicMock(), MagicMock()]

        with (
            patch("pdfplumber.open", return_value=MagicMock(__enter__=MagicMock(return_value=mock_doc))),
            pytest.raises(ValueError, match="start_page"),
        ):
            extract_tables.discover_tables(
                pdf_path=fake_pdf,
                start_page=5,
                end_page=2,
                preview_dir=tmp_path / "preview",
            )

    def test_guess_heading_from_page(self) -> None:
        """Tests guessing headings from page text and table rows."""
        page_mock = MagicMock()
        page_mock.extract_text.return_value = (
            "2025-26 Review: Regular Season\n"
            "REGULAR-SEASON STATISTICS\n"
            "TEAMS\n"
            "– 7 –"
        )
        table_rows: list[list[str | None]] = [["", "TEAMS", "", ""]]
        heading = extract_tables.guess_heading_from_page(page_mock, table_rows)
        assert "2025-26 Review: Regular Season" in heading
        assert "REGULAR-SEASON STATISTICS" in heading

    def test_extract_page_tables_fallback(self) -> None:
        """Tests fallback from empty lines strategy to text strategy."""
        page_mock = MagicMock()
        # Line extraction returns nothing
        page_mock.extract_tables.side_effect = [
            [],  # First call (default lines)
            [[["Col1", "Col2"], ["Val1", "Val2"]]],  # Second call (text strategy)
        ]

        tables = extract_tables.extract_page_tables(page_mock)
        assert len(tables) == 1
        assert tables[0] == [["Col1", "Col2"], ["Val1", "Val2"]]

    def test_discover_tables_writes_csv_and_report(self, tmp_path: Path) -> None:
        """Tests discover_tables creates preview CSV files and table_report.md."""
        fake_pdf = tmp_path / "sample.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 dummy")

        page_mock = MagicMock()
        page_mock.extract_text.return_value = "Heading 1\nSubheading 2"
        page_mock.extract_tables.return_value = [
            [["Team", "Wins", "Losses"], ["Celtics", "64", "18"], ["Knicks", "50", "32"]],
        ]

        mock_doc = MagicMock()
        mock_doc.pages = [page_mock]

        preview_dir = tmp_path / "previews"
        report_file = tmp_path / "report.md"

        with patch("pdfplumber.open", return_value=MagicMock(__enter__=MagicMock(return_value=mock_doc))):
            results = extract_tables.discover_tables(
                pdf_path=fake_pdf,
                start_page=1,
                end_page=1,
                output_report=report_file,
                preview_dir=preview_dir,
            )

        assert len(results) == 1
        assert results[0]["page"] == 1
        assert results[0]["table_num"] == 1
        assert results[0]["row_count"] == 3
        assert results[0]["col_count"] == 3

        # Verify CSV content
        csv_file = preview_dir / "1_1.csv"
        assert csv_file.exists()
        with open(csv_file, encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert reader[0] == ["Team", "Wins", "Losses"]
            assert reader[1] == ["Celtics", "64", "18"]

        # Verify report
        assert report_file.exists()
        report_content = report_file.read_text(encoding="utf-8")
        assert "# Discovered Tables Report" in report_content
        assert "1_1.csv" in report_content


class TestTableCleaning:
    """Tests for table cleaning functions."""

    def test_clean_number(self) -> None:
        """Tests numeric conversion with commas, decimals, and null markers."""
        assert extract_tables.clean_number("1,234") == 1234
        assert extract_tables.clean_number("29.6") == 29.6
        assert extract_tables.clean_number("-5") == -5
        assert extract_tables.clean_number("  42.0  ") == 42.0
        assert extract_tables.clean_number("–") is None
        assert extract_tables.clean_number("N/A") is None
        assert extract_tables.clean_number(None) is None

    def test_split_w_l(self) -> None:
        """Tests splitting Win-Loss strings."""
        assert extract_tables.split_w_l("60-22") == (60, 22)
        assert extract_tables.split_w_l("4-1") == (4, 1)
        assert extract_tables.split_w_l("24- 17") == (24, 17)
        assert extract_tables.split_w_l("invalid") == (None, None)
        assert extract_tables.split_w_l("") == (None, None)

    def test_split_team_coach(self) -> None:
        """Tests splitting Team (Coach) strings."""
        team, coach = extract_tables.split_team_coach("Philadelphia (Eddie Gottlieb)")
        assert team == "Philadelphia"
        assert coach == "Eddie Gottlieb"

        team, coach = extract_tables.split_team_coach("*Boston (Doc Rivers)")
        assert team == "Boston"
        assert coach == "Doc Rivers"

        team, coach = extract_tables.split_team_coach("No Coach Mentioned")
        assert team == "No Coach Mentioned"
        assert coach == ""

    def test_drop_repeated_headers(self) -> None:
        """Tests removing repeated header rows."""
        data = {
            "Player": ["LeBron James", "Player", "Stephen Curry"],
            "Total": [21, "Total", 11],
        }
        df = pd.DataFrame(data)
        cleaned = extract_tables.drop_repeated_headers(df, ["Player", "Total"])
        assert len(cleaned) == 2
        assert list(cleaned["Player"]) == ["LeBron James", "Stephen Curry"]


def _word(text: str, x0: float, top: float) -> dict[str, Any]:
    return {"text": text, "x0": x0, "x1": x0 + 5 * len(text), "top": top, "bottom": top + 8}


def _fake_page(words: list[dict[str, Any]], text: str = "") -> MagicMock:
    page = MagicMock()
    page.extract_words.return_value = words
    page.extract_text.return_value = text
    page.crop.return_value = page
    page.height = 648.0
    return page


def _finals_spec() -> dict[str, Any]:
    with open(Path(__file__).parent / "tables.yaml", encoding="utf-8") as f:
        spec: dict[str, Any] = yaml.safe_load(f)["tables"]["nba_finals"]
    return spec


class TestLineExtraction:
    """Tests for line clustering, skipped-line reporting, and the Finals line regex."""

    def test_cluster_lines_keeps_words_across_rounding_boundary_together(self) -> None:
        # round(4.4/3)*3 == 3 but round(4.6/3)*3 == 6: bucketing would split this one printed line.
        words = [_word("5", 60, 4.6), _word("LeBron", 10, 4.4), _word("Next", 10, 20.0)]
        lines = extract_tables.cluster_lines(words)
        assert [[w["text"] for w in line] for line in lines] == [["LeBron", "5"], ["Next"]]

    def test_stacked_blocks_parse_rows_and_report_skipped_lines(self) -> None:
        words = [
            _word("Player", 30, 50), _word("Total", 88, 50),
            _word("LeBron", 30, 60), _word("James", 60, 60.9), _word("21", 88, 60),
            _word("13", 107, 60), _word("4", 120, 60), _word("4", 135, 60),
            _word("Mitch", 160, 50), _word("Richmond", 185, 50), _word("5", 215, 50),
            _word("0", 232, 50), _word("3", 245, 50), _word("2", 260, 50),
            _word("Broken", 160, 60), _word("Row", 190, 60), _word("5", 215, 60), _word("x", 232, 60),
            _word("Footer", 30, 640),
        ]
        blocks = [{"x_range": [30, 155]}, {"x_range": [155, 280]}]
        rows, skipped = extract_tables._extract_stacked_blocks(_fake_page(words), 111, blocks, [45, 630])

        assert [r["player"] for r in rows] == ["LeBron James", "Mitch Richmond"]
        assert rows[0] == {"player": "LeBron James", "total": 21, "first_team": 13, "second_team": 4, "third_team": 4}
        assert [(s.page, s.text) for s in skipped] == [(111, "Player Total"), (111, "Broken Row 5 x")]

    def test_word_columns_skip_lines_with_unassigned_words(self) -> None:
        splits = {"pick": [0, 20], "player": [20, 100]}
        words = [
            _word("Pick", 0, 10), _word("Player", 20, 10),
            _word("1.", 0, 20), _word("AJ", 20, 20), _word("Dybantsa", 35, 20),
            _word("2.", 0, 30), _word("Stray", 150, 30),
        ]
        rows, skipped = extract_tables._extract_word_columns(_fake_page(words), 52, splits)

        assert rows == [{"pick": 1, "player": "AJ Dybantsa"}]
        assert [s.reason.split(":")[0] for s in skipped] == ["no leading pick number", "words outside every column"]

    def test_finals_regex_handles_cross_month_dates_and_spaced_asterisk(self) -> None:
        spec = _finals_spec()
        text = (
            "Year Dates Winning Team (Coach) Losing Team (Coach) Games\n"
            "2020 Sept. 30-Oct. 11 LA Lakers (Frank Vogel) Miami (Erik Spoelstra) 4-2\n"
            "1959 April 4-9 * Boston (Red Auerbach) Minneapolis (John Kundla) 4-0\n"
            "1955 March 31-April 10 *Syracuse (Al Cervi) *Fort Wayne (Charley Eckman) 4-3\n"
        )
        rows, skipped = extract_tables._extract_line_regex(
            _fake_page([], text), 180, spec["line_pattern"], spec["line_fields"]
        )

        assert [(r["year"], r["dates"], r["winner_team"]) for r in rows] == [
            ("2020", "Sept. 30-Oct. 11", "LA Lakers"),
            ("1959", "April 4-9", "* Boston"),
            ("1955", "March 31-April 10", "*Syracuse"),
        ]
        assert [s.text for s in skipped] == ["Year Dates Winning Team (Coach) Losing Team (Coach) Games"]

    def test_asterisk_flags_become_columns(self, tmp_path: Path) -> None:
        spec = _finals_spec()
        spec["pages"] = [{"page": 1}]
        text = "1959 April 4-9 * Boston (Red Auerbach) Minneapolis (John Kundla) 4-0\n"
        pdf = MagicMock()
        pdf.__enter__.return_value.pages = [_fake_page([], text)]

        with patch("extract_tables.pdfplumber.open", return_value=pdf):
            result = extract_tables.load_and_clean_table(tmp_path / "x.pdf", "nba_finals", spec)

        row = result.df.iloc[0]
        assert (row["winner_team"], row["winner_best_record"]) == ("Boston", 1)
        assert (row["loser_team"], row["loser_best_record"]) == ("Minneapolis", 0)
        assert row["year"] == 1959 and row["loser_games"] == 0


class TestValidatedLoad:
    """The database is written only when every validation check passes."""

    @staticmethod
    def _tables() -> dict[str, extract_tables.ExtractedTable]:
        df = pd.DataFrame({"pick": [1, 2], "player": ["AJ Dybantsa", "Darryn Peterson"]})
        return {"draft_2026": extract_tables.ExtractedTable(df)}

    def test_refuses_to_write_when_any_check_fails(self, tmp_path: Path) -> None:
        db = tmp_path / "nba stats.db"
        with (
            patch("extract_tables.load_tables_from_spec", return_value=self._tables()),
            patch("validate_tables.read_source_text", return_value={}),
            patch("extract_tables.save_tables_to_sqlite") as mock_save,
        ):
            # Real checks: 2 draft rows instead of 60, and two pilot tables missing.
            results = extract_tables.load_validated("guide.pdf", "tables.yaml", db)

        assert not all(c.passed for c in results)
        mock_save.assert_not_called()
        assert not db.exists()

    def test_cli_load_exits_nonzero_and_does_not_write(self, tmp_path: Path) -> None:
        db = tmp_path / "nba stats.db"
        argv = ["extract_tables.py", "load", "--pdf", "guide.pdf", "--db-path", str(db)]
        with (
            patch("sys.argv", argv),
            patch("extract_tables.load_tables_from_spec", return_value=self._tables()),
            patch("validate_tables.read_source_text", return_value={}),
            pytest.raises(SystemExit, match="database was not written"),
        ):
            extract_tables.main()
        assert not db.exists()

    def test_writes_when_all_checks_pass(self, tmp_path: Path) -> None:
        db = tmp_path / "nba stats.db"
        passing = [Check("draft_2026", "ok", True)]
        with (
            patch("extract_tables.load_tables_from_spec", return_value=self._tables()),
            patch("validate_tables.read_source_text", return_value={}),
            patch("validate_tables.run_checks", return_value=passing),
        ):
            results = extract_tables.load_validated("guide.pdf", "tables.yaml", db)

        assert results == passing
        with closing(sqlite3.connect(db)) as conn:
            assert extract_tables.table_row_counts(conn) == {"draft_2026": 2}

    def test_replace_overwrites_previous_rows(self, tmp_path: Path) -> None:
        db = tmp_path / "nba stats.db"
        extract_tables.save_tables_to_sqlite({"t": pd.DataFrame({"a": [1, 2, 3]})}, db)
        extract_tables.save_tables_to_sqlite({"t": pd.DataFrame({"a": [9]})}, db)
        with closing(sqlite3.connect(db)) as conn:
            assert conn.execute("SELECT a FROM t").fetchall() == [(9,)]


class TestSchemaText:
    def test_lists_tables_columns_types_and_two_sample_rows(self, tmp_path: Path) -> None:
        db = tmp_path / "dir with space" / "nba stats.db"
        db.parent.mkdir()
        df = pd.DataFrame({"year": [1947, 1948, 1949], "winner_team": ["Philadelphia", "Baltimore", "Minneapolis"]})
        extract_tables.save_tables_to_sqlite({"nba_finals": df}, db)

        text = extract_tables.schema_text(db)

        assert "Table: nba_finals (3 rows)" in text
        assert "Columns: year INTEGER, winner_team TEXT" in text
        assert "Philadelphia" in text and "Baltimore" in text
        assert "Minneapolis" not in text

    def test_missing_db_does_not_create_file(self, tmp_path: Path) -> None:
        db = tmp_path / "missing.db"
        assert "does not exist" in extract_tables.schema_text(db)
        assert not db.exists()
