"""Acronym expansion for search queries (runtime module; reads acronyms.yaml, no pipeline imports).

Users type "Finals MVP", "ROTY", "GSW" or "steals"; the guide prints "Finals Most Valuable Player",
"Rookie of the Year", "Golden State" and the column "STL". expand() rewrites a query into the printed
wording so the embedding sees both forms; prompt_line() gives the agent the same mapping for SQL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ACRONYMS_PATH: Path = Path(__file__).with_name("acronyms.yaml")
DIRECTIONS: frozenset[str] = frozenset({"replace", "add_acronym"})
RRF_K: int = 60
_WORD = r"A-Za-z0-9_%\-"  # hyphen and % count as word characters, so "three-point" never matches "point"


@dataclass(frozen=True)
class Entry:
    """One acronyms.yaml entry."""

    category: str
    canonical: str
    acronyms: tuple[str, ...]
    ambiguous: bool = False
    unverified: bool = False
    direction: str = "replace"


@dataclass(frozen=True)
class Expansion:
    """An expanded query and the terms (typed form and canonical) that triggered it."""

    query: str
    terms: tuple[str, ...]


def _entry(category: str, raw: dict[str, Any]) -> Entry:
    canonical = str(raw.get("canonical", "")).strip()
    acronyms = tuple(str(a).strip() for a in raw.get("acronyms") or [] if str(a).strip())
    direction = str(raw.get("direction", "replace"))
    if not canonical or not acronyms:
        raise ValueError(f"acronyms.yaml: {category} entry needs 'canonical' and 'acronyms': {raw!r}")
    if direction not in DIRECTIONS:
        raise ValueError(f"acronyms.yaml: direction must be one of {sorted(DIRECTIONS)}, got {direction!r}")
    return Entry(category, canonical, acronyms, bool(raw.get("ambiguous", False)),
                 bool(raw.get("unverified", False)), direction)


@lru_cache(maxsize=4)
def load_entries(path: Path = ACRONYMS_PATH) -> tuple[Entry, ...]:
    """All entries, in file order.

    Raises:
        ValueError: If an entry is malformed.
    """
    with open(path, encoding="utf-8") as f:
        data: dict[str, list[dict[str, Any]]] = yaml.safe_load(f) or {}
    return tuple(_entry(category, raw) for category, items in data.items() for raw in items or [])


def _pattern(term: str) -> str:
    body = r"\s+".join(re.escape(part) for part in term.split())
    plural = "s?" if term[-1].isalpha() else ""
    return rf"(?<![{_WORD}]){body}{plural}(?![{_WORD}])"


@lru_cache(maxsize=4)
def _matchers(path: Path = ACRONYMS_PATH) -> tuple[re.Pattern[str], dict[str, tuple[Entry, bool]]]:
    """One alternation regex (longest term first) and term (lowercased) -> (entry, typed_canonical)."""
    lookup: dict[str, tuple[Entry, bool]] = {}
    for entry in load_entries(path):
        if entry.ambiguous:
            continue
        for acronym in entry.acronyms:
            lookup.setdefault(acronym.lower(), (entry, False))
        if entry.direction == "add_acronym":
            lookup.setdefault(entry.canonical.lower(), (entry, True))
    terms = sorted(lookup, key=len, reverse=True)
    regex = re.compile("|".join(f"({_pattern(t)})" for t in terms), re.IGNORECASE)
    return regex, lookup


def _term_for(matched: str, lookup: dict[str, tuple[Entry, bool]]) -> str:
    key = " ".join(matched.lower().split())
    if key in lookup:
        return key
    return key[:-1]  # trailing plural "s"


def expand(query: str, path: Path = ACRONYMS_PATH) -> Expansion | None:
    """Rewrites acronyms into the guide's printed wording; None if nothing matched.

    Whole words, case-insensitive, longest match first ("Finals MVP" before "MVP"), a trailing plural "s"
    allowed. Acronyms become their canonical name; a stat word (steals, blocks, ...) gets its printed
    column acronym inserted after it. Ambiguous entries are never expanded.
    """
    regex, lookup = _matchers(path)
    terms: list[str] = []

    def replace(match: re.Match[str]) -> str:
        text = match.group(0)
        entry, typed_canonical = lookup[_term_for(text, lookup)]
        terms.extend([text, entry.canonical])
        if entry.direction == "add_acronym":
            terms.append(entry.acronyms[0])
        if entry.direction == "add_acronym":
            word = text if typed_canonical else entry.canonical
            return f"{word} {entry.acronyms[0]}"
        return entry.canonical

    expanded = regex.sub(replace, query)
    if not terms:
        return None
    return Expansion(query=expanded, terms=tuple(dict.fromkeys(terms)))


def expand_query(query: str) -> str | None:
    """The enriched query, or None if no acronym or stat word matched."""
    result = expand(query)
    return result.query if result is not None else None


def mentions_any(text: str, terms: tuple[str, ...]) -> bool:
    """True if `text` contains any term, ignoring case and whitespace (PDF text often has stray spaces)."""
    squashed = re.sub(r"\s+", "", text).lower()
    return any(re.sub(r"\s+", "", t).lower() in squashed for t in terms)


def rrf_merge(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal rank fusion: score(id) = sum of 1 / (k + rank); ties keep first-appearance order."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    order = list(dict.fromkeys(item for ranking in rankings for item in ranking))
    return sorted(order, key=lambda item: -scores[item])


def prompt_line(categories: tuple[str, ...] = ("awards", "teams"), path: Path = ACRONYMS_PATH) -> str:
    """Compact "ACRONYM = printed name" list for the system prompt (unverified entries left out)."""
    parts: list[str] = []
    for entry in load_entries(path):
        if entry.category in categories and not entry.unverified:
            parts.append(f"{'/'.join(entry.acronyms)} = {entry.canonical}")
    return "; ".join(dict.fromkeys(parts))


def unverified_entries(path: Path = ACRONYMS_PATH) -> list[Entry]:
    return [e for e in load_entries(path) if e.unverified]
