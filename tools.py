"""LangChain tools for the NBA Guide agent: semantic search over the guide and read-only SQL.

Page numbers are PDF page indexes (the printed page number is one lower), so every
user-facing citation is written "PDF p. N".
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.errors import ChromaError, NotFoundError
from google import genai
from google.genai.errors import APIError
from langchain_core.tools import tool

import config
import sqlsafe
from embed_data import embed_texts, to_chroma_vectors

logger = logging.getLogger(__name__)

SEARCH_K: int = 5


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk: its PDF page index, cosine distance, and text."""

    page: int
    distance: float
    text: str


class RetrievalError(RuntimeError):
    """The vector store is missing, empty, or could not be queried."""


@lru_cache(maxsize=1)
def get_embed_client() -> genai.Client:
    """Client for query embeddings, using the PAID embedding key (never the free LLM key)."""
    return genai.Client(api_key=config.get_embed_api_key(required=True))


def get_collection(
    data_dir: str | Path | None = None,
    name: str = config.DEFAULT_COLLECTION,
) -> chromadb.Collection:
    """Opens the existing guide collection without creating anything on disk.

    Raises:
        RetrievalError: If the directory or collection does not exist.
    """
    data_path = Path(data_dir if data_dir is not None else config.DATA_DIR).resolve()
    if not data_path.is_dir():
        raise RetrievalError(f"Vector store '{data_path.name}' not found. Run embed_data.py first.")
    try:
        return chromadb.PersistentClient(path=str(data_path)).get_collection(name=name)
    except (NotFoundError, ValueError) as e:
        raise RetrievalError(f"Collection '{name}' not found. Run embed_data.py first.") from e


def retrieve(
    query: str,
    k: int = SEARCH_K,
    collection: chromadb.Collection | None = None,
    client: genai.Client | None = None,
) -> list[Hit]:
    """Embeds `query` (RETRIEVAL_QUERY) and returns the `k` nearest chunks, closest first.

    Raises:
        ValueError: If the query is blank or k < 1.
        RetrievalError: If the collection is missing or empty.
    """
    if not query.strip():
        raise ValueError("Search query is empty.")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}.")

    collection = collection if collection is not None else get_collection()
    count = collection.count()
    if count == 0:
        raise RetrievalError(f"Collection '{collection.name}' is empty. Run embed_data.py first.")

    vectors = embed_texts(
        client=client if client is not None else get_embed_client(),
        texts=[query],
        task_type="RETRIEVAL_QUERY",
    )
    res = collection.query(
        query_embeddings=to_chroma_vectors(vectors),
        n_results=min(k, count),
        include=["documents", "metadatas", "distances"],
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits: list[Hit] = []
    for doc, meta, dist in zip(docs, metas, dists):
        page = meta.get("page") if meta else None
        hits.append(Hit(page=int(page) if isinstance(page, (int, float)) else 0, distance=float(dist), text=doc or ""))
    return hits


def format_hits(hits: list[Hit]) -> str:
    """Formats hits as delimited passages the LLM can cite by PDF page."""
    if not hits:
        return "No matching passages found in the guide."
    return "\n\n".join(f"--- PDF p. {h.page} (distance {h.distance:.3f}) ---\n{h.text.strip()}" for h in hits)


@tool
def search_guide(query: str) -> str:
    """Semantic search over the 2026-27 NBA Guide text. Use for narrative questions (history, records,
    rules, biographies, explanations) and anything the SQL tables do not cover.

    Args:
        query: A focused natural-language search query, e.g. "most points scored in a single game".

    Returns:
        Up to 5 passages, each headed by its page as "PDF p. N", or an error message.
    """
    try:
        return format_hits(retrieve(query, k=SEARCH_K))
    except (ValueError, RetrievalError) as e:
        return f"SEARCH ERROR: {e}"
    except APIError as e:
        logger.warning("Embedding API error during search: %s", e)
        return f"SEARCH ERROR: embedding service failed ({e.code}). Try again later."
    except ChromaError as e:
        logger.warning("Chroma error during search: %s", e)
        return f"SEARCH ERROR: vector store query failed ({type(e).__name__})."


@tool
def run_sql(sql: str) -> str:
    """Run one read-only SQLite SELECT against the stats tables (schema in the system prompt). Use for
    counting, ranking, filtering, and aggregation.

    Args:
        sql: A single SELECT or WITH ... SELECT statement. No semicolons, no comments. At most 50 rows
            are returned, so use ORDER BY / LIMIT / aggregates.

    Returns:
        A markdown table of results followed by a "Sources: table = PDF pp. ..." footer for the tables the
        query reads, or an error message explaining what to fix.
    """
    try:
        result = sqlsafe.safe_query(sql, db_path=config.DB_PATH)
    except sqlsafe.UnsafeSQLError as e:
        return f"SQL REJECTED: {e}"
    except sqlsafe.QueryTimeoutError as e:
        return f"SQL TIMEOUT: {e}"
    except FileNotFoundError as e:
        return f"SQL ERROR: {e}"
    except sqlite3.Error as e:
        return f"SQL ERROR: {e}. Check table and column names against the schema and retry."
    try:
        footer = sqlsafe.source_footer(sql, db_path=config.DB_PATH)
    except (sqlsafe.UnsafeSQLError, sqlsafe.QueryTimeoutError, sqlite3.Error) as e:
        logger.warning("Could not build the source-pages footer: %s", e)
        footer = ""
    return f"{result}\n\n{footer}" if footer else result
