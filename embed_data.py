"""Ingestion script for NBA Guide PDF.

Extracts text from a PDF, splits into structure-preserving chunks,
generates embeddings using Google Gemini (gemini-embedding-001),
and stores them in a local ChromaDB vector store under ./data.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import re
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pypdf import PdfReader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from config import DEFAULT_COLLECTION, EMBED_DIM, EMBED_MODEL, get_embed_api_key

# Load environment variables (.env)
load_dotenv()

logger = logging.getLogger(__name__)

# Constants
RETRYABLE_STATUS_CODES: set[int] = {429, 500, 503, 504}
SENTENCE_END_RE: re.Pattern[str] = re.compile(r"(?<=[a-z]{2}|\d{2})\.(?=\s+[A-Z])")


def embed_texts(
    client: genai.Client,
    texts: list[str],
    task_type: str,
    model: str = EMBED_MODEL,
    dim: int = EMBED_DIM,
    max_retries: int = 6,
) -> list[list[float]]:
    """Generates embeddings using Google Gemini with resilient retry backoff.

    Args:
        client: Google GenAI client instance.
        texts: List of document or query texts to embed.
        task_type: Embedding task type (e.g. 'RETRIEVAL_DOCUMENT' or 'RETRIEVAL_QUERY').
        model: Model identifier.
        dim: Output vector dimensionality.
        max_retries: Maximum number of retry attempts for retryable errors.

    Returns:
        List of embedding vectors, one for each input text.

    Raises:
        APIError: If non-retryable error occurs or retries are exhausted.
    """
    if not texts:
        return []

    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=dim,
    )

    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=model,
                contents=texts,
                config=config,
            )
            return [emb.values for emb in response.embeddings]
        except APIError as e:
            if e.code not in RETRYABLE_STATUS_CODES or attempt == max_retries - 1:
                raise
            base = 35.0 if e.code == 429 else 2.0
            delay = min(120.0, base * (1.5**attempt)) + random.random()
            tqdm.write(
                f"[Retry {attempt + 1}/{max_retries}] API error {e.code}: "
                f"{e.message or 'Request failed'}. Retrying in {delay:.2f}s..."
            )
            time.sleep(delay)


def generate_chunk_id(source: str, page: int, text: str) -> str:
    """Computes a deterministic content-based ID for a chunk.

    Args:
        source: Source document filename.
        page: Page number where chunk originates.
        text: Text content of the chunk.

    Returns:
        A 16-character SHA1 hex digest string.
    """
    content = f"{source}|{page}|{text}".encode()
    return hashlib.sha1(content).hexdigest()[:16]


def normalize(text: str) -> str:
    """Normalizes document whitespace while preserving paragraph and line structure.

    Collapses runs of spaces and tabs to a single space, and collapses 3 or
    more newlines to a single blank line (2 newlines). Single newlines are preserved.

    Args:
        text: Raw text string.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of horizontal whitespace (spaces/tabs) to a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Strip spaces surrounding newlines
    text = re.sub(r" ?\n ?", "\n", text)
    # Collapse 3+ newlines to a blank line (2 newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Splits text into chunks preserving structure and sentence boundaries.

    Prefers split points in order:
    1. Paragraph break (\\n\\n)
    2. Single newline (\\n)
    3. Sentence end (avoiding abbreviations/initials)
    4. Space

    Args:
        text: Text to chunk.
        chunk_size: Maximum character length per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.

    Returns:
        List of non-empty text chunks.
    """
    text = normalize(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        if text_len - start <= chunk_size:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        window_end = start + chunk_size
        search_start = start + chunk_size // 2

        split_pos = -1

        # 1. Paragraph break (\n\n)
        p_break = text.rfind("\n\n", search_start, window_end)
        if p_break != -1:
            split_pos = p_break

        # 2. Single newline (\n)
        if split_pos == -1:
            nl = text.rfind("\n", search_start, window_end)
            if nl != -1:
                split_pos = nl

        # 3. Sentence end
        if split_pos == -1:
            sub = text[:window_end]
            matches = [m for m in SENTENCE_END_RE.finditer(sub) if m.end() >= search_start]
            if matches:
                split_pos = matches[-1].end()

        # 4. Space
        if split_pos == -1:
            sp = text.rfind(" ", search_start, window_end)
            if sp != -1:
                split_pos = sp

        # 5. Pathological fallback
        if split_pos == -1:
            split_pos = window_end

        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward, stepping back by overlap
        new_start = max(split_pos - chunk_overlap, start + 1)

        # Advance start to the next word boundary so chunks never begin mid-word
        if (
            new_start < text_len
            and new_start > 0
            and not text[new_start - 1].isspace()
            and not text[new_start].isspace()
        ):
            next_space = text.find(" ", new_start)
            if next_space != -1 and next_space < split_pos:
                while next_space < text_len and text[next_space].isspace():
                    next_space += 1
                new_start = next_space

        # Guarantee strict forward progress to avoid infinite loops
        start = max(new_start, start + 1)

    return chunks


def extract_and_chunk_pdf(
    pdf_path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    """Reads PDF and returns a list of chunk dictionaries with metadata.

    Args:
        pdf_path: Path to the target PDF file.
        chunk_size: Maximum character length per chunk.
        chunk_overlap: Character overlap between consecutive chunks.
        max_pages: Optional maximum number of pages to process.

    Returns:
        List of dicts with 'id', 'text', and 'metadata'.

    Raises:
        FileNotFoundError: If the PDF file cannot be located.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    logger.info("Opening PDF: %s (%.1f MB)...", pdf_path.name, file_size_mb)
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    process_pages = min(total_pages, max_pages) if max_pages else total_pages

    logger.info("Extracting and chunking %d of %d pages...", process_pages, total_pages)

    documents: list[dict[str, Any]] = []
    global_chunk_idx = 0
    skipped_pages = 0

    for page_idx in tqdm(range(process_pages), desc="Reading Pages", unit="page"):
        page_num = page_idx + 1
        try:
            page_text = reader.pages[page_idx].extract_text() or ""
        except (ValueError, TypeError, KeyError, IndexError) as e:
            logger.warning("Could not extract text from page %d: %s", page_num, e)
            continue

        page_text = page_text.strip()
        # Skip pages with virtually no text (e.g. blank or full image pages)
        if len(page_text) < 40:
            skipped_pages += 1
            logger.debug("Skipped page %d (< 40 characters).", page_num)
            continue

        page_chunks = chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for chunk_idx_in_page, chunk in enumerate(page_chunks):
            chunk_id = generate_chunk_id(pdf_path.name, page_num, chunk)
            documents.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_num,
                        "chunk_index": global_chunk_idx,
                    },
                }
            )
            global_chunk_idx += 1

    if process_pages > 0:
        skip_ratio = skipped_pages / process_pages
        logger.info(
            "Pages skipped (< 40 chars): %d / %d (%.1f%%)",
            skipped_pages,
            process_pages,
            skip_ratio * 100,
        )
        if skip_ratio > 0.20:
            logger.warning(
                "High skip rate: %.1f%% of pages had under 40 characters. "
                "This document may be a scanned PDF requiring OCR.",
                skip_ratio * 100,
            )

    logger.info("Total extracted chunks: %d", len(documents))
    return documents


def store_in_chromadb(
    documents: list[dict[str, Any]],
    data_dir: str | Path = "./data",
    collection_name: str = DEFAULT_COLLECTION,
    batch_size: int = 50,
    reset: bool = False,
    delay: float = 1.0,
    skip_stale_cleanup: bool = False,
    client: genai.Client | None = None,
) -> chromadb.Collection:
    """Stores documents and embeddings in ChromaDB with safe re-ingestion.

    Args:
        documents: List of chunk dictionaries to ingest.
        data_dir: Directory where ChromaDB files are stored.
        collection_name: Name of the Chroma collection.
        batch_size: Number of chunks per embedding batch and upsert.
        reset: If True, deletes existing collection (without removing files on disk).
        delay: Seconds to sleep between batches to avoid rate limits.
        skip_stale_cleanup: If True, skips deleting chunks not in documents (used when --max-pages is set).
        client: Optional pre-configured Google GenAI client.

    Returns:
        The target Chroma collection instance.
    """
    data_path = Path(data_dir).resolve()
    data_path.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(data_path))

    if reset:
        try:
            chroma_client.delete_collection(name=collection_name)
            logger.info("Deleted existing collection '%s'.", collection_name)
        except (NotFoundError, ValueError):
            logger.debug("Collection '%s' did not exist to delete.", collection_name)

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"model": EMBED_MODEL, "dimension": EMBED_DIM, "hnsw:space": "cosine"},
    )

    if not documents:
        logger.info("No documents provided to store.")
        return collection

    pdf_name = documents[0]["metadata"]["source"]
    new_id_set = {doc["id"] for doc in documents}

    # Fetch existing IDs for this source only
    existing_result = collection.get(where={"source": pdf_name}, include=[])
    existing_ids = set(existing_result["ids"]) if existing_result and "ids" in existing_result else set()

    # Clean up stale chunks for this source if document content or chunking changed.
    # We skip this cleanup when ingesting a partial subset (e.g. --max-pages is set).
    if not skip_stale_cleanup:
        stale_ids = existing_ids - new_id_set
        if stale_ids:
            logger.info("Deleting %d stale chunks for source '%s'...", len(stale_ids), pdf_name)
            collection.delete(ids=list(stale_ids))
            existing_ids -= stale_ids
    else:
        logger.debug("Partial ingestion (--max-pages is set): skipping stale chunk cleanup.")

    # Skip chunks already embedded and persisted
    documents_to_embed = [doc for doc in documents if doc["id"] not in existing_ids]
    if existing_ids:
        logger.info(
            "Resuming: %d chunks already in collection for '%s'. %d chunks remaining to embed.",
            len(existing_ids),
            pdf_name,
            len(documents_to_embed),
        )

    total_to_embed = len(documents_to_embed)
    if total_to_embed == 0:
        logger.info("All chunks for '%s' are already embedded and up to date.", pdf_name)
        return collection

    if client is None:
        api_key = get_embed_api_key(required=True)
        client = genai.Client(api_key=api_key)

    logger.info(
        "Generating Gemini embeddings (%s, d=%d) and adding %d chunks in batches of %d...",
        EMBED_MODEL,
        EMBED_DIM,
        total_to_embed,
        batch_size,
    )

    with tqdm(total=total_to_embed, desc="Embedding & Storing", unit="chunk") as pbar:
        for i in range(0, total_to_embed, batch_size):
            batch = documents_to_embed[i : i + batch_size]
            b_ids = [doc["id"] for doc in batch]
            b_texts = [doc["text"] for doc in batch]
            b_metas = [doc["metadata"] for doc in batch]

            b_embeddings = embed_texts(
                client=client,
                texts=b_texts,
                task_type="RETRIEVAL_DOCUMENT",
                model=EMBED_MODEL,
                dim=EMBED_DIM,
            )

            collection.upsert(
                ids=b_ids,
                documents=b_texts,
                metadatas=b_metas,
                embeddings=b_embeddings,
            )
            pbar.update(len(batch))
            if delay > 0 and (i + batch_size) < total_to_embed:
                time.sleep(delay)

    final_count = collection.count()
    logger.info("Embedding complete! Total records in '%s': %d", collection_name, final_count)
    return collection


def verify_database(
    client: genai.Client,
    collection: chromadb.Collection,
    sample_query: str = "Who holds the record for most points in a game?",
) -> None:
    """Demonstrates vector retrieval by embedding a query and searching collection.

    Args:
        client: Google GenAI client.
        collection: Chroma collection instance.
        sample_query: Query string to test.
    """
    logger.info("Running verification query: %r", sample_query)
    q_embeddings = embed_texts(
        client=client,
        texts=[sample_query],
        task_type="RETRIEVAL_QUERY",
        model=EMBED_MODEL,
        dim=EMBED_DIM,
    )

    results = collection.query(
        query_embeddings=q_embeddings,
        n_results=2,
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        page = meta.get("page", "?")
        tqdm.write(f"\n[Result #{rank}] (Page {page}, Distance: {dist:.4f})")
        tqdm.write(f"{doc[:300]}...")


def main() -> None:
    """CLI entrypoint for embed_data."""
    parser = argparse.ArgumentParser(description="Embed NBA Guide PDF into ChromaDB using Gemini embeddings.")
    parser.add_argument(
        "--pdf",
        type=str,
        default="2026-27-NBA-Guide.pdf",
        help="Path to the PDF file (default: 2026-27-NBA-Guide.pdf)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to persist ChromaDB (default: ./data)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help=f"Name of the ChromaDB collection (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for embedding and storing (default: 50)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Character length per chunk (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap characters between chunks (default: 200)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit number of pages to process (useful for testing, default: all pages)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to pause between batches to avoid rate limits (default: 1.0)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing data in data directory before embedding",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    with logging_redirect_tqdm():
        docs = extract_and_chunk_pdf(
            pdf_path=args.pdf,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_pages=args.max_pages,
        )

        api_key = get_embed_api_key(required=False)
        client = genai.Client(api_key=api_key) if api_key else None

        collection = store_in_chromadb(
            documents=docs,
            data_dir=args.data_dir,
            collection_name=args.collection,
            batch_size=args.batch_size,
            reset=args.reset,
            delay=args.delay,
            skip_stale_cleanup=(args.max_pages is not None),
            client=client,
        )

        if collection and collection.count() > 0 and client:
            verify_database(client, collection)


if __name__ == "__main__":
    main()
