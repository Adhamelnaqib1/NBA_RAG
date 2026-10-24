"""Pure metric functions for the evaluation harness (no I/O, no API calls).

Retrieval metrics work on a ranked list of PDF pages with duplicates removed (several chunks can come from
one page; only the first occurrence counts). A page is relevant if it is in the gold page set.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass


def dedupe_pages(pages: Iterable[int]) -> list[int]:
    """Pages in first-seen order, each once."""
    seen: set[int] = set()
    out: list[int] = []
    for p in pages:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def first_relevant_rank(ranked: Sequence[int], relevant: Iterable[int]) -> int | None:
    """1-based rank of the first relevant page, or None."""
    rel = set(relevant)
    return next((i for i, p in enumerate(ranked, start=1) if p in rel), None)


def hit_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """1.0 if any relevant page is in the top k."""
    rank = first_relevant_rank(ranked[:k], relevant)
    return 1.0 if rank is not None else 0.0


def recall_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fraction of the relevant pages found in the top k (equals hit@k when one page is relevant)."""
    rel = set(relevant)
    if not rel:
        raise ValueError("recall is undefined with no relevant pages")
    return len(rel & set(ranked[:k])) / len(rel)


def reciprocal_rank(ranked: Sequence[int], relevant: Iterable[int]) -> float:
    rank = first_relevant_rank(ranked, relevant)
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Binary-relevance nDCG@k: gain 1 for each relevant page, discounted by log2(rank + 1)."""
    rel = set(relevant)
    if not rel:
        raise ValueError("nDCG is undefined with no relevant pages")
    dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(ranked[:k], start=1) if p in rel)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal


@dataclass(frozen=True)
class Estimate:
    """A mean with a percentile-bootstrap confidence interval."""

    mean: float
    low: float
    high: float
    n: int


def bootstrap_ci(
    values: Sequence[float],
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    stat: Callable[[Sequence[float]], float] | None = None,
) -> Estimate:
    """Mean of `values` (or `stat`) with a percentile bootstrap CI; deterministic for a given seed."""
    fn = stat or (lambda xs: sum(xs) / len(xs))
    n = len(values)
    if n == 0:
        return Estimate(0.0, 0.0, 0.0, 0)
    rng = random.Random(seed)
    stats = sorted(fn([values[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    tail = (1.0 - confidence) / 2
    low = stats[int(tail * resamples)]
    high = stats[min(resamples - 1, int((1.0 - tail) * resamples))]
    return Estimate(fn(values), low, high, n)
