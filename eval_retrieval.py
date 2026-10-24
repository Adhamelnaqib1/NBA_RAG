"""Retrieval eval: hit@k, recall@k, MRR and nDCG@k of tools.search over the gold set. No LLM calls.

Query embeddings are cached on disk (eval_results/embed_cache.jsonl), keyed by model, dimension, task type and
text, so each distinct query is embedded once ever per model. tools.search runs unchanged: only its embed_texts is
swapped for the cached one, so the evaluated path is the production path. --backend gemini embeds on the PAID key
(asks first); --backend bge runs BAAI/bge-base-en-v1.5 locally against ./data_bge (no key, no cost).

    python eval_retrieval.py --backend bge --check     # validate the gold set, show the plan; no embedding calls
    python eval_retrieval.py --backend bge             # local run
    python eval_retrieval.py --backend gemini          # paid run for cache misses (asks for confirmation)
    python eval_retrieval.py --offline                 # cache only: fails on a miss instead of embedding
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from google import genai

import acronyms
import config
import embed_data
import eval_gold
import eval_metrics
import tools

logger = logging.getLogger(__name__)

DEFAULT_KS: tuple[int, ...] = (4, 8, 10)
DEFAULT_OUT_DIR = Path("eval_results")
CACHE_NAME = "embed_cache.jsonl"
QUERY_TASK = "RETRIEVAL_QUERY"
METRICS: tuple[str, ...] = ("hit", "recall", "mrr", "ndcg")

SearchFn = Callable[[str, int], list[int]]
"""(query, k) -> PDF pages of the top-k retrieved chunks, closest first (duplicates allowed)."""


class CacheMiss(RuntimeError):
    """An embedding was needed that is not in the cache, in offline mode."""


class EmbeddingCache:
    """On-disk cache with the same call signature as embed_data.embed_texts.

    Misses are embedded in one batch with the real embed_texts and appended to the JSONL file; in offline mode a
    miss raises CacheMiss instead.
    """

    def __init__(self, path: Path, offline: bool = False) -> None:
        self.path = path
        self.offline = offline
        self.live_texts = 0
        self._vectors: dict[str, list[float]] = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for n, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        self._vectors[str(row["key"])] = [float(x) for x in row["vector"]]
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                        raise ValueError(f"{path}:{n}: corrupt cache line ({e})") from e

    @staticmethod
    def key(text: str, task_type: str, model: str = config.EMBED_MODEL, dim: int = config.EMBED_DIM) -> str:
        return hashlib.sha256(f"{model}|{dim}|{task_type}|{text}".encode()).hexdigest()

    def __contains__(self, key: object) -> bool:
        return key in self._vectors

    def __len__(self) -> int:
        return len(self._vectors)

    def embed_texts(
        self,
        client: genai.Client | None,
        texts: list[str],
        task_type: str,
        model: str = config.EMBED_MODEL,
        dim: int = config.EMBED_DIM,
        max_retries: int = 6,
        device: str | None = None,
    ) -> list[list[float]]:
        keys = [self.key(t, task_type, model, dim) for t in texts]
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in self._vectors))
        if missing:
            if self.offline:
                raise CacheMiss(f"{len(missing)} query embedding(s) not cached, e.g. {missing[0]!r}")
            vectors = embed_data.embed_texts(client=client, texts=missing, task_type=task_type, model=model,
                                             dim=dim, max_retries=max_retries, device=device)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                for text, vector in zip(missing, vectors):
                    k = self.key(text, task_type, model, dim)
                    self._vectors[k] = vector
                    f.write(json.dumps({"key": k, "model": model, "dim": dim, "task": task_type, "text": text,
                                        "vector": vector}) + "\n")
            self.live_texts += len(missing)
        return [self._vectors[k] for k in keys]


@contextmanager
def cached_embeddings(cache: EmbeddingCache) -> Iterator[None]:
    """Routes tools.search's query embeddings through `cache` for the duration of the block."""
    original = tools.embed_texts
    tools.embed_texts = cache.embed_texts
    try:
        yield
    finally:
        tools.embed_texts = original


def planned_queries(items: Sequence[eval_gold.GoldItem], mode: str) -> list[str]:
    """Every distinct text tools.search may embed for these items: the question, plus its acronym expansion
    unless mode is off (in fallback mode the expansion is embedded only when needed, so this is an upper bound)."""
    texts: list[str] = []
    for item in items:
        texts.append(item.question)
        expansion = acronyms.expand(item.question) if mode != "off" else None
        if expansion is not None:
            texts.append(expansion.query)
    return list(dict.fromkeys(texts))


def score(pages: Sequence[int], relevant: Sequence[int], k: int) -> dict[str, Any]:
    ranked = eval_metrics.dedupe_pages(pages)
    return {
        "pages": ranked,
        "rank": eval_metrics.first_relevant_rank(ranked[:k], relevant),
        "hit": eval_metrics.hit_at_k(ranked, relevant, k),
        "recall": eval_metrics.recall_at_k(ranked, relevant, k),
        "mrr": eval_metrics.reciprocal_rank(ranked[:k], relevant),
        "ndcg": eval_metrics.ndcg_at_k(ranked, relevant, k),
    }


def summarize(rows: Sequence[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    """Mean and 95% bootstrap CI of each metric, overall and per item type."""
    groups: dict[str, list[dict[str, Any]]] = {"all": list(rows)}
    for row in rows:
        groups.setdefault(row["type"], []).append(row)
    return {
        name: {m: asdict(eval_metrics.bootstrap_ci([r[m] for r in group], seed=seed)) for m in METRICS}
        for name, group in groups.items()
    }


def evaluate(items: Sequence[eval_gold.GoldItem], search: SearchFn, ks: Sequence[int]) -> dict[str, Any]:
    """Runs `search` once per item and k (fused rankings depend on k, so smaller k is not a prefix of larger)."""
    by_k: dict[str, Any] = {}
    for k in ks:
        rows = []
        for item in items:
            row = {"id": item.id, "type": item.type, "relevant": list(item.retrieval_pages)}
            row.update(score(search(item.question, k), item.retrieval_pages, k))
            rows.append(row)
        by_k[str(k)] = {"summary": summarize(rows), "items": rows}
    return by_k


def report(results: dict[str, Any]) -> str:
    """Markdown table: one row per k and item group, each metric as mean [95% CI]."""
    lines = ["| k | group | n | hit@k | recall@k | MRR | nDCG@k |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, block in results.items():
        for group, metrics in block["summary"].items():
            cells = [f"{metrics[m]['mean']:.3f} [{metrics[m]['low']:.2f}, {metrics[m]['high']:.2f}]" for m in METRICS]
            lines.append(f"| {k} | {group} | {metrics['hit']['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def live_search(mode: str, client: genai.Client | None, spec: config.EmbedSpec) -> SearchFn:
    """tools.search against the backend's real collection; embeddings go through whatever tools.embed_texts is."""
    collection = tools.get_collection(spec.data_dir, spec.collection)

    def run(query: str, k: int) -> list[int]:
        result = tools.search(query, k=k, collection=collection, client=client, mode=mode, spec=spec)
        return [hit.page for hit in result.hits]

    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval eval (hit@k, recall@k, MRR, nDCG@k). No LLM calls.")
    parser.add_argument("--gold", type=Path, default=eval_gold.DEFAULT_GOLD, help="Gold set YAML")
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS), help="k values (default: 4 8 10)")
    parser.add_argument("--mode", choices=config.ACRONYM_MODES, default=None,
                        help="Acronym mode (default: ACRONYM_MODE or 'fallback')")
    parser.add_argument("--backend", choices=config.EMBED_BACKENDS, default=None,
                        help="Embedding backend and its store (default: EMBED_BACKEND or gemini)")
    parser.add_argument("--types", nargs="+", choices=eval_gold.TYPES, default=list(eval_gold.RETRIEVAL_TYPES),
                        help="Item types to score (default: narrative mixed)")
    parser.add_argument("--include-unverified", action="store_true", help="Also score items with verified: false")
    parser.add_argument("--label", default=None, help="Run label for the output file (default: the mode)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Results directory")
    parser.add_argument("--check", action="store_true", help="Validate and print the plan; no API calls")
    parser.add_argument("--offline", action="store_true", help="Use cached embeddings only; fail on a miss")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation before live embedding calls")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if any(k < 1 for k in args.k):
        print("Error: every k must be at least 1", file=sys.stderr)
        return 2
    try:
        mode = args.mode or config.get_acronym_mode()
        spec = config.get_embed_spec(args.backend)
        items = eval_gold.select(eval_gold.load_gold(args.gold), tuple(args.types), args.include_unverified)
        cache = EmbeddingCache(args.out_dir / CACHE_NAME, offline=args.offline)
    except (OSError, ValueError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if not items:
        print("Error: no items selected (are they verified?)", file=sys.stderr)
        return 1

    texts = planned_queries(items, mode)
    misses = sum(EmbeddingCache.key(t, QUERY_TASK, spec.model, spec.dim) not in cache for t in texts)
    counts = ", ".join(f"{t}={sum(i.type == t for i in items)}" for t in args.types)
    where = "computed locally (no key, no cost)" if spec.local else "on the PAID key"
    print(f"{len(items)} items ({counts}); k={args.k}; mode={mode}; backend={spec.backend} ({spec.model}, "
          f"{spec.data_dir}/{spec.collection})")
    print(f"{len(texts)} distinct query texts, {len(texts) - misses} cached; "
          f"at most {misses} embeddings {where}, 0 LLM calls.")
    if args.check:
        return 0
    # In fallback mode the count above is an upper bound, so --offline runs anyway: a query that really needs an
    # uncached embedding raises CacheMiss below.
    if (misses and not spec.local and not args.offline and not args.yes
            and input(f"Embed up to {misses} queries on the PAID key now? [y/N] ").strip().lower() != "y"):
        print("Cancelled.")
        return 1

    # A local backend never needs a client. For Gemini with nothing to embed, the cache answers every call, so a
    # placeholder stands in: it must not be None, or tools.search would build a real client (and need the key).
    client: genai.Client | None = None
    if not spec.local:
        client = tools.get_embed_client() if misses and not args.offline else cast(genai.Client, object())
    try:
        with cached_embeddings(cache):
            results = evaluate(items, live_search(mode, client, spec), args.k)
    except (CacheMiss, tools.RetrievalError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    label = args.label or f"{spec.backend}_{mode}"
    out = args.out_dir / f"retrieval_{label}.json"
    payload = {
        "label": label,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "gold_sha256": _file_sha256(args.gold),
        "config": {"mode": mode, "ks": args.k, "types": args.types, "include_unverified": args.include_unverified,
                   "backend": spec.backend, "collection": spec.collection, "embed_model": spec.model,
                   "embed_dim": spec.dim, "reranker": None},
        "live_embeddings": cache.live_texts,
        "results": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(report(results))
    print(f"\n{cache.live_texts} live embedding(s). Results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
