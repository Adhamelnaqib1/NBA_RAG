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

import acronyms
import config
import sqlsafe
from embed_data import embed_texts, to_chroma_vectors

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class Hit:
    """One retrieved chunk: its PDF page index, cosine distance, text and Chroma id."""

    page: int
    distance: float
    text: str
    id: str = ""


@dataclass(frozen=True)
class SearchResult:
    """Hits after acronym handling, and every query actually sent to the vector store."""

    hits: list[Hit]
    queries: list[str]


class RetrievalError(RuntimeError):
    """The vector store is missing, empty, or could not be queried."""


@lru_cache(maxsize=1)
def get_embed_client() -> genai.Client:
    """Client for query embeddings, using the PAID embedding key (never the free LLM key)."""
    return genai.Client(api_key=config.get_embed_api_key(required=True))


def get_collection(
    data_dir: str | Path | None = None,
    name: str | None = None,
) -> chromadb.Collection:
    """Opens the existing guide collection without creating anything on disk (defaults: EMBED_BACKEND's store).

    Raises:
        RetrievalError: If the directory or collection does not exist.
    """
    spec = config.get_embed_spec()
    name = name or spec.collection
    data_path = Path(data_dir if data_dir is not None else spec.data_dir).resolve()
    if not data_path.is_dir():
        raise RetrievalError(f"Vector store '{data_path.name}' not found. Run embed_data.py first.")
    try:
        return chromadb.PersistentClient(path=str(data_path)).get_collection(name=name)
    except (NotFoundError, ValueError) as e:
        raise RetrievalError(f"Collection '{name}' not found. Run embed_data.py first.") from e


def _query_chroma(collection: chromadb.Collection, vector: list[float], n: int) -> list[Hit]:
    res = collection.query(
        query_embeddings=to_chroma_vectors([vector]),
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    hits: list[Hit] = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        page = meta.get("page") if meta else None
        hits.append(Hit(page=int(page) if isinstance(page, (int, float)) else 0, distance=float(dist),
                        text=doc or "", id=str(cid)))
    return hits


def merge_hits(rankings: list[list[Hit]], k: int) -> list[Hit]:
    """Reciprocal rank fusion (k=60) over several hit lists, de-duplicated by chunk id, top k."""
    by_id: dict[str, Hit] = {}
    for ranking in rankings:
        for hit in ranking:
            by_id.setdefault(hit.id, hit)
    order = acronyms.rrf_merge([[h.id for h in ranking] for ranking in rankings])
    return [by_id[i] for i in order[:k]]


def search(
    query: str,
    k: int | None = None,
    collection: chromadb.Collection | None = None,
    client: genai.Client | None = None,
    mode: str | None = None,
    spec: config.EmbedSpec | None = None,
) -> SearchResult:
    """Embeds `query` (RETRIEVAL_QUERY) and returns the `k` nearest chunks (default SEARCH_K), closest first.

    `spec` is the embedding backend the collection was built with (default: EMBED_BACKEND); a Gemini client is
    only created for a Gemini backend.

    With an acronym match (acronyms.expand) and ACRONYM_MODE:
      both: the original and expanded queries are embedded in ONE embed_texts call, each searched, and the
            results merged by reciprocal rank fusion;
      fallback: the original is searched; only if no passage mentions a matched term is the expanded query
            embedded and searched too, then merged;
      off: no expansion.
    Without a match exactly one query is made.

    Raises:
        ValueError: If the query is blank, k < 1, or ACRONYM_MODE is invalid.
        RetrievalError: If the collection is missing or empty.
    """
    if not query.strip():
        raise ValueError("Search query is empty.")
    if k is None:
        k = config.get_search_k()
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}.")
    mode = mode if mode is not None else config.get_acronym_mode()

    collection = collection if collection is not None else get_collection()
    count = collection.count()
    if count == 0:
        raise RetrievalError(f"Collection '{collection.name}' is empty. Run embed_data.py first.")
    spec = spec or config.get_embed_spec()
    if client is None and not spec.local:
        client = get_embed_client()
    n = min(k, count)

    def embed(texts: list[str]) -> list[list[float]]:
        return embed_texts(client=client, texts=texts, task_type="RETRIEVAL_QUERY", model=spec.model, dim=spec.dim)

    expansion = acronyms.expand(query) if mode != "off" else None
    if expansion is not None and mode == "both":
        original, expanded = embed([query, expansion.query])
        rankings = [_query_chroma(collection, original, n), _query_chroma(collection, expanded, n)]
        return SearchResult(merge_hits(rankings, n), [query, expansion.query])

    [vector] = embed([query])
    hits = _query_chroma(collection, vector, n)
    if expansion is not None and mode == "fallback" and not any(
        acronyms.mentions_any(h.text, expansion.terms) for h in hits
    ):
        [vector2] = embed([expansion.query])
        rankings = [hits, _query_chroma(collection, vector2, n)]
        return SearchResult(merge_hits(rankings, n), [query, expansion.query])
    return SearchResult(hits, [query])


def retrieve(
    query: str,
    k: int | None = None,
    collection: chromadb.Collection | None = None,
    client: genai.Client | None = None,
    mode: str | None = None,
    spec: config.EmbedSpec | None = None,
) -> list[Hit]:
    """The hits of search(); kept for callers that only need the passages."""
    return search(query, k=k, collection=collection, client=client, mode=mode, spec=spec).hits


def searched_line(queries: list[str]) -> str:
    """'Searched: "q1" + "q2"' — one line; whitespace collapsed so it can never look like a passage header."""
    return "Searched: " + " + ".join('"' + " ".join(q.split()).replace('"', "'") + '"' for q in queries)


def format_hits(hits: list[Hit]) -> str:
    """Formats hits as delimited passages the LLM can cite by PDF page."""
    if not hits:
        return "No matching passages found in the guide."
    return "\n\n".join(f"--- PDF p. {h.page} (distance {h.distance:.3f}) ---\n{h.text.strip()}" for h in hits)


@tool
def search_guide(query: str, k: int | None = None) -> str:
    """Semantic search over the 2026-27 NBA Guide text. Use for narrative questions (history, records,
    rules, biographies, explanations) and anything the SQL tables do not cover.

    Args:
        query: The user's own words, e.g. "who was finals mvp in 2021"; acronyms are expanded for you.
        k: Optional number of passages (1-10); omit it to use the default.

    Returns:
        A "Searched: ..." line (acronyms are expanded automatically), then up to k passages (default 8), each
        headed by its page as "PDF p. N"; or an error message.
    """
    try:
        if k is not None:
            k = max(1, min(int(k), config.MAX_SEARCH_K))
        result = search(query, k=k)
        return f"{searched_line(result.queries)}\n\n{format_hits(result.hits)}"
    except (ValueError, RetrievalError) as e:
        return f"SEARCH ERROR: {e}"
    except APIError as e:
        logger.warning("Embedding API error during search: %s", e)
        return f"SEARCH ERROR: embedding service failed ({e.code}). Try again later."
    except ChromaError as e:
        logger.warning("Chroma error during search: %s", e)
        return f"SEARCH ERROR: vector store query failed ({type(e).__name__})."
    except OSError as e:
        logger.warning("Local embedding model could not be loaded: %s", e)
        return "SEARCH ERROR: local embedding model could not be loaded."


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
