"""Unit tests for tools.py (offline: Gemini is mocked, Chroma and SQLite live in tmp_path)."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import MagicMock

import chromadb
import pytest
from google.genai.errors import APIError
from langchain_core.tools import BaseTool

import config
import tools
from embed_data import EMBED_DIM, to_chroma_vectors
from tools import Hit, RetrievalError, retrieve, run_sql, search_guide


def _gemini(vector: list[float] | None = None) -> MagicMock:
    client = MagicMock()
    values = vector if vector is not None else [1.0] + [0.0] * (EMBED_DIM - 1)
    client.models.embed_content.return_value = MagicMock(embeddings=[MagicMock(values=values)])
    return client


def _unit(i: int) -> list[float]:
    v = [0.0] * EMBED_DIM
    v[i] = 1.0
    return v


@pytest.fixture
def collection(tmp_path: Path) -> chromadb.Collection:
    col = chromadb.PersistentClient(path=str(tmp_path / "chroma data")).create_collection(
        name=config.DEFAULT_COLLECTION, configuration={"hnsw": {"space": "cosine"}}
    )
    col.add(
        ids=["a", "b", "c"],
        documents=["Wilt scored 100 points.", "Draft lottery rules.", "Finals history."],
        metadatas=[{"page": 300, "source": "g.pdf"}, {"page": 52, "source": "g.pdf"}, {"page": 179, "source": "g.pdf"}],
        embeddings=to_chroma_vectors([_unit(0), _unit(1), _unit(2)]),
    )
    return col


class TestRetrieve:
    def test_returns_hits_closest_first_using_retrieval_query(self, collection: chromadb.Collection) -> None:
        client = _gemini()
        hits = retrieve("most points in a game", k=2, collection=collection, client=client)

        assert len(hits) == 2
        assert (hits[0].page, hits[0].text) == (300, "Wilt scored 100 points.")
        assert hits[0].distance == pytest.approx(0.0, abs=1e-5)
        assert hits[0].distance < hits[1].distance
        kwargs = client.models.embed_content.call_args.kwargs
        assert kwargs["config"].task_type == "RETRIEVAL_QUERY"
        assert kwargs["config"].output_dimensionality == EMBED_DIM

    def test_k_larger_than_collection_is_capped(self, collection: chromadb.Collection) -> None:
        assert len(retrieve("x", k=50, collection=collection, client=_gemini())) == 3

    def test_empty_collection_raises(self, tmp_path: Path) -> None:
        empty = chromadb.PersistentClient(path=str(tmp_path)).create_collection(name="empty")
        with pytest.raises(RetrievalError, match="empty"):
            retrieve("x", collection=empty, client=_gemini())

    def test_missing_dir_raises_and_creates_nothing(self, tmp_path: Path) -> None:
        missing = tmp_path / "no data"
        with pytest.raises(RetrievalError, match="not found"):
            tools.get_collection(missing)
        assert not missing.exists()

    def test_missing_collection_raises(self, tmp_path: Path) -> None:
        chromadb.PersistentClient(path=str(tmp_path)).create_collection(name="other")
        with pytest.raises(RetrievalError, match="Collection 'nba_guide"):
            tools.get_collection(tmp_path)

    @pytest.mark.parametrize(("query", "k"), [("  ", 5), ("ok", 0)])
    def test_bad_arguments_raise(self, collection: chromadb.Collection, query: str, k: int) -> None:
        with pytest.raises(ValueError):
            retrieve(query, k=k, collection=collection, client=_gemini())

    def test_embed_client_uses_paid_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_EMBED_API_KEY", "paid-key")
        monkeypatch.setenv("GEMINI_LLM_API_KEY", "free-key")
        mock_client_cls = MagicMock()
        monkeypatch.setattr(tools.genai, "Client", mock_client_cls)
        tools.get_embed_client.cache_clear()
        try:
            tools.get_embed_client()
        finally:
            tools.get_embed_client.cache_clear()
        mock_client_cls.assert_called_once_with(api_key="paid-key")


class TestSearchGuideTool:
    def test_is_langchain_tool(self) -> None:
        assert isinstance(search_guide, BaseTool)
        assert isinstance(run_sql, BaseTool)
        assert search_guide.name == "search_guide"
        assert run_sql.name == "run_sql"
        assert set(search_guide.args) == {"query"}
        assert set(run_sql.args) == {"sql"}

    def test_formats_passages_with_pdf_pages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tools, "retrieve", lambda q, k: [Hit(52, 0.1234, " Draft text. ")])
        out = search_guide.invoke({"query": "draft"})
        assert out == "--- PDF p. 52 (distance 0.123) ---\nDraft text."

    def test_no_hits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tools, "retrieve", lambda q, k: [])
        assert "No matching passages" in search_guide.invoke({"query": "x"})

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (RetrievalError("Collection 'x' is empty."), "SEARCH ERROR: Collection 'x' is empty."),
            (ValueError("Search query is empty."), "SEARCH ERROR: Search query is empty."),
            (APIError(429, {"error": {"message": "quota"}}, None), "embedding service failed (429)"),
            (chromadb.errors.InternalError("boom"), "vector store query failed"),
        ],
    )
    def test_errors_become_strings(self, monkeypatch: pytest.MonkeyPatch, error: Exception, expected: str) -> None:
        def boom(q: str, k: int) -> list[Hit]:
            raise error

        monkeypatch.setattr(tools, "retrieve", boom)
        assert expected in search_guide.invoke({"query": "x"})


class TestRunSqlTool:
    @pytest.fixture(autouse=True)
    def db(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        path = tmp_path / "nba stats.db"
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.execute("CREATE TABLE draft_2026 (pick INTEGER, player TEXT)")
            conn.execute("INSERT INTO draft_2026 VALUES (1, 'AJ Dybantsa')")
        monkeypatch.setattr(config, "DB_PATH", path)
        return path

    def test_returns_markdown(self) -> None:
        out = run_sql.invoke({"sql": "SELECT pick, player FROM draft_2026"})
        assert out.splitlines()[2] == "| 1 | AJ Dybantsa |"

    @pytest.mark.parametrize(
        ("sql", "prefix"),
        [
            ("DELETE FROM draft_2026", "SQL REJECTED:"),
            ("SELECT 1; SELECT 2", "SQL REJECTED:"),
            ("SELECT nope FROM draft_2026", "SQL ERROR: no such column"),
            ("SELEC 1", "SQL REJECTED:"),
        ],
    )
    def test_errors_become_strings(self, sql: str, prefix: str) -> None:
        assert run_sql.invoke({"sql": sql}).startswith(prefix)

    def test_timeout_becomes_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = tools.sqlsafe.safe_query

        def fast_timeout(sql: str, db_path: Path) -> str:
            return original(sql, db_path, timeout_s=0.1)

        monkeypatch.setattr(tools.sqlsafe, "safe_query", fast_timeout)
        cte = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
        assert run_sql.invoke({"sql": cte}).startswith("SQL TIMEOUT:")

    def test_missing_db_becomes_string(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "DB_PATH", tmp_path / "gone.db")
        assert run_sql.invoke({"sql": "SELECT 1"}).startswith("SQL ERROR: Database 'gone.db' not found")
