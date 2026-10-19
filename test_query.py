"""Unit tests for query.py (offline; Chroma and Gemini are mocked)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from config import EMBED_DIM
from query import query_vector_store


def _gemini_client() -> MagicMock:
    client = MagicMock()
    client.models.embed_content.return_value = MagicMock(embeddings=[MagicMock(values=[0.05] * EMBED_DIM)])
    return client


def _run_with_results(tmp_path: Path, results: dict[str, Any]) -> MagicMock:
    collection = MagicMock()
    collection.count.return_value = 3
    collection.query.return_value = results
    with patch("query.chromadb.PersistentClient") as mock_persistent:
        mock_persistent.return_value.get_collection.return_value = collection
        query_vector_store("Who won in 1990?", k=2, data_dir=tmp_path, client=_gemini_client())
    return collection


class TestQueryVectorStore:
    """Tests for query_vector_store result handling."""

    @pytest.mark.parametrize(
        "results",
        [
            {"ids": [[]], "documents": None, "metadatas": None, "distances": None},
            {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]},
            {},
        ],
    )
    def test_empty_or_none_results_print_no_matches(
        self, tmp_path: Path, results: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run_with_results(tmp_path, results)
        assert "No matching documents found." in capsys.readouterr().out

    def test_hits_are_printed_and_query_is_float32(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        results = {
            "ids": [["a"]],
            "documents": [["Detroit beat Portland 4-1."]],
            "metadatas": [[{"page": 179, "source": "guide.pdf"}]],
            "distances": [[0.1234]],
        }
        collection = _run_with_results(tmp_path, results)

        out = capsys.readouterr().out
        assert "Page: 179" in out
        assert "Distance: 0.1234" in out
        sent = collection.query.call_args.kwargs["query_embeddings"]
        assert isinstance(sent, np.ndarray)
        assert sent.dtype == np.float32
        assert collection.query.call_args.kwargs["n_results"] == 2
