"""Unit tests for extract_tables.py discovery functionality."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import extract_tables


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
        table_rows = [["", "TEAMS", "", ""]]
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
        import pandas as pd  # type: ignore[import-untyped]
        data = {
            "Player": ["LeBron James", "Player", "Stephen Curry"],
            "Total": [21, "Total", 11],
        }
        df = pd.DataFrame(data)
        cleaned = extract_tables.drop_repeated_headers(df, ["Player", "Total"])
        assert len(cleaned) == 2
        assert list(cleaned["Player"]) == ["LeBron James", "Stephen Curry"]
