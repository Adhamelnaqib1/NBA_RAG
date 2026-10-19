"""Query CLI for NBA Guide ChromaDB vector store.

Embeds a user question with Google Gemini (task_type='RETRIEVAL_QUERY')
and retrieves the most relevant chunks from ChromaDB.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from dotenv import load_dotenv
from google import genai

from config import DEFAULT_COLLECTION, EMBED_DIM, EMBED_MODEL, get_embed_api_key
from embed_data import embed_texts, to_chroma_vectors

load_dotenv()

logger = logging.getLogger(__name__)


def query_vector_store(
    question: str,
    k: int = 5,
    data_dir: str | Path = "./data",
    collection_name: str = DEFAULT_COLLECTION,
    client: genai.Client | None = None,
) -> None:
    """Queries the ChromaDB collection with an embedded question and prints hits.

    Args:
        question: The user query string.
        k: Number of top matches to return.
        data_dir: Path to ChromaDB persistent storage directory.
        collection_name: Target collection name.
        client: Optional pre-configured Google GenAI client.

    Raises:
        ValueError: If API key is missing or collection is empty/missing.
    """
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        sys.exit(f"Error: Database directory '{data_path}' does not exist. Run embed_data.py first.")

    chroma_client = chromadb.PersistentClient(path=str(data_path))

    try:
        collection = chroma_client.get_collection(name=collection_name)
    except (NotFoundError, ValueError):
        sys.exit(
            f"Error: Collection '{collection_name}' does not exist in '{data_path}'. "
            f"Please run ingestion first using embed_data.py."
        )

    count = collection.count()
    if count == 0:
        sys.exit(f"Error: Collection '{collection_name}' exists but contains 0 records.")

    if client is None:
        try:
            api_key = get_embed_api_key(required=True)
        except ValueError as err:
            sys.exit(f"Error: {err}")
        client = genai.Client(api_key=api_key)

    query_embeddings = embed_texts(
        client=client,
        texts=[question],
        task_type="RETRIEVAL_QUERY",
        model=EMBED_MODEL,
        dim=EMBED_DIM,
    )

    results = collection.query(
        query_embeddings=to_chroma_vectors(query_embeddings),
        n_results=min(k, count),
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    if not documents:
        print("No matching documents found.")
        return

    print(f"\nTop {len(documents)} results for query: {question!r}\n" + "-" * 60)
    for rank, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        page = meta.get("page", "?")
        source = meta.get("source", "unknown")
        print(f"[Rank #{rank}] Source: {source} | Page: {page} | Distance: {dist:.4f}")
        preview = doc[:300].strip().replace("\n", " ")
        print(f"{preview}...")
        print("-" * 60)


def main() -> None:
    """CLI entry point for query.py."""
    parser = argparse.ArgumentParser(description="Query the NBA Guide vector store using Gemini embeddings.")
    parser.add_argument("question", type=str, help="Question or search query string")
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbors to retrieve (default: 5)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Path to the ChromaDB directory (default: ./data)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help=f"Target ChromaDB collection (default: {DEFAULT_COLLECTION})",
    )
    args = parser.parse_args()

    query_vector_store(
        question=args.question,
        k=args.k,
        data_dir=args.data_dir,
        collection_name=args.collection,
    )


if __name__ == "__main__":
    main()
