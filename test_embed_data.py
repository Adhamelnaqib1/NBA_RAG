"""Unit tests for embed_data.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from google.genai.errors import APIError

from embed_data import (
    DEFAULT_COLLECTION,
    EMBED_DIM,
    chunk_text,
    embed_texts,
    generate_chunk_id,
    normalize,
    store_in_chromadb,
)


class TestChunking:
    """Tests for text normalization and chunking logic."""

    def test_normalize_whitespace_and_newlines(self) -> None:
        raw = "Line 1   with   spaces\t\tand tabs.\n\n\n\nLine 2.\nLine 3."
        norm = normalize(raw)
        assert "   " not in norm
        assert "\t" not in norm
        assert "\n\n\n" not in norm
        assert "Line 1 with spaces and tabs.\n\nLine 2.\nLine 3." == norm

    def test_chunk_text_respects_max_size(self) -> None:
        text = "The quick brown fox jumps over the lazy dog.\n\n" * 30
        chunk_size = 200
        overlap = 40
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=overlap)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= chunk_size
            assert len(chunk) > 0

    def test_chunk_text_keeps_newlines(self) -> None:
        text = "Paragraph one with some details.\n\nParagraph two with more details.\nLine continuation."
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert "\n\n" in chunks[0]
        assert "\nLine continuation." in chunks[0]

    def test_chunk_text_never_starts_mid_word(self) -> None:
        text = (
            "Basketball is a team sport in which two teams, most commonly of five players each, "
            "opposing one another on a rectangular court, compete with the primary objective "
            "of shooting a basketball through the defender's hoop."
        )
        chunks = chunk_text(text, chunk_size=70, chunk_overlap=25)
        assert len(chunks) > 1
        for chunk in chunks:
            # First character must not be whitespace and should be the beginning of a word in text
            assert not chunk[0].isspace()
            # If the chunk appears in text, check that its start in text isn't immediately preceded by a letter
            pos = text.find(chunk)
            if pos > 0:
                assert text[pos - 1].isspace(), f"Chunk starts mid-word: '{chunk[:15]}...'"

    def test_chunk_text_does_not_split_initials_or_abbreviations(self) -> None:
        sentence_1 = "The Portland Trail Blazers traded guard C.J. McCollum to the Pelicans."
        sentence_2 = "He joined forward P.J. Tucker and guard T.J. McConnell in the division."
        sentence_3 = "The team finished No. 1 in defensive rating while playing in St. Louis."
        sentence_4 = "Legendary star Dr. J attended the game and praised their effort."
        full_text = f"{sentence_1} {sentence_2} {sentence_3} {sentence_4}"

        # Choose a chunk size that forces splitting
        chunks = chunk_text(full_text, chunk_size=100, chunk_overlap=20)
        recombined = " ".join(chunks)

        # Ensure initials were not cut in half across chunk splits
        assert "C.J." in recombined
        assert "P.J." in recombined
        assert "T.J." in recombined
        assert "No. 1" in recombined
        assert "St. Louis" in recombined
        assert "Dr. J" in recombined

        for c in chunks:
            assert not c.endswith("C.")
            assert not c.endswith("P.")
            assert not c.endswith("T.")
            assert not c.startswith("J. McCollum")
            assert not c.startswith("J. Tucker")

    def test_chunk_text_pathological_inputs(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   \n\n\t  \n  ") == []

        # Long unbroken string without any spaces or newlines
        unbroken = "a" * 2500
        chunks = chunk_text(unbroken, chunk_size=500, chunk_overlap=100)
        assert len(chunks) >= 5
        for c in chunks:
            assert len(c) <= 500


class TestChunkId:
    """Tests for deterministic chunk ID generation."""

    def test_chunk_id_deterministic(self) -> None:
        id1 = generate_chunk_id("guide.pdf", 12, "Sample chunk text.")
        id2 = generate_chunk_id("guide.pdf", 12, "Sample chunk text.")
        assert id1 == id2
        assert len(id1) == 16

    def test_chunk_id_varies_with_components(self) -> None:
        base = generate_chunk_id("guide.pdf", 12, "Sample chunk text.")
        diff_text = generate_chunk_id("guide.pdf", 12, "Different text.")
        diff_page = generate_chunk_id("guide.pdf", 13, "Sample chunk text.")
        diff_source = generate_chunk_id("other.pdf", 12, "Sample chunk text.")

        assert base != diff_text
        assert base != diff_page
        assert base != diff_source


class TestEmbeddingRetry:
    """Tests for embed_texts retry and error handling."""

    def test_embed_texts_retries_on_429_503_then_succeeds(self) -> None:
        mock_client = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * EMBED_DIM
        mock_response = MagicMock(embeddings=[mock_embedding])

        err_429 = APIError(429, "Resource exhausted", None)
        err_503 = APIError(503, "Service unavailable", None)

        mock_client.models.embed_content.side_effect = [err_429, err_503, mock_response]

        with patch("embed_data.time.sleep") as mock_sleep:
            result = embed_texts(
                client=mock_client,
                texts=["NBA champions"],
                task_type="RETRIEVAL_DOCUMENT",
                max_retries=4,
            )

        assert len(result) == 1
        assert len(result[0]) == EMBED_DIM
        assert mock_client.models.embed_content.call_count == 3
        assert mock_sleep.call_count == 2

    @pytest.mark.parametrize("status_code", [400, 401, 403, 404])
    def test_embed_texts_raises_immediately_on_non_retryable_status(self, status_code: int) -> None:
        mock_client = MagicMock()
        non_retryable_err = APIError(status_code, f"Client error {status_code}", None)
        mock_client.models.embed_content.side_effect = non_retryable_err

        with patch("embed_data.time.sleep") as mock_sleep:
            with pytest.raises(APIError) as exc_info:
                embed_texts(
                    client=mock_client,
                    texts=["Query"],
                    task_type="RETRIEVAL_QUERY",
                    max_retries=5,
                )

            assert exc_info.value.code == status_code
            mock_sleep.assert_not_called()
        assert mock_client.models.embed_content.call_count == 1


class TestChromaReset:
    """Tests for safe --reset behavior in store_in_chromadb."""

    def test_reset_preserves_unrelated_files_on_disk(self, tmp_path: pytest.TempPathFactory) -> None:
        unrelated_file = tmp_path / "user_notes.txt"
        unrelated_file.write_text("Do not delete me!")

        mock_client = MagicMock()
        mock_emb = MagicMock(values=[0.05] * EMBED_DIM)
        mock_client.models.embed_content.return_value = MagicMock(embeddings=[mock_emb])

        dummy_docs = [
            {
                "id": "abc1234567890def",
                "text": "Lakers defeated the Celtics.",
                "metadata": {"source": "nba.pdf", "page": 1, "chunk_index": 0},
            }
        ]

        # Ingest with reset=True
        store_in_chromadb(
            documents=dummy_docs,
            data_dir=tmp_path,
            collection_name=DEFAULT_COLLECTION,
            reset=True,
            client=mock_client,
            delay=0.0,
        )

        assert unrelated_file.exists()
        assert unrelated_file.read_text() == "Do not delete me!"

        # Re-run with reset=True again to ensure safe deletion of collection only
        store_in_chromadb(
            documents=dummy_docs,
            data_dir=tmp_path,
            collection_name=DEFAULT_COLLECTION,
            reset=True,
            client=mock_client,
            delay=0.0,
        )

        assert unrelated_file.exists()
        assert unrelated_file.read_text() == "Do not delete me!"

    def test_skip_stale_cleanup_preserves_other_chunks(self, tmp_path: pytest.TempPathFactory) -> None:
        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = lambda model, contents, config: MagicMock(
            embeddings=[MagicMock(values=[0.05] * EMBED_DIM) for _ in contents]
        )

        chunk_1 = {
            "id": "chunk_page_1",
            "text": "Page 1 content",
            "metadata": {"source": "nba.pdf", "page": 1, "chunk_index": 0},
        }
        chunk_2 = {
            "id": "chunk_page_2",
            "text": "Page 2 content",
            "metadata": {"source": "nba.pdf", "page": 2, "chunk_index": 1},
        }

        # First ingest both chunks
        col = store_in_chromadb(
            documents=[chunk_1, chunk_2],
            data_dir=tmp_path,
            collection_name=DEFAULT_COLLECTION,
            reset=True,
            client=mock_client,
            delay=0.0,
        )
        assert col.count() == 2

        # Second ingestion with only chunk_1, but skip_stale_cleanup=True (as when --max-pages is set)
        col = store_in_chromadb(
            documents=[chunk_1],
            data_dir=tmp_path,
            collection_name=DEFAULT_COLLECTION,
            reset=False,
            client=mock_client,
            delay=0.0,
            skip_stale_cleanup=True,
        )
        # chunk_2 must NOT be deleted because skip_stale_cleanup=True
        assert col.count() == 2
