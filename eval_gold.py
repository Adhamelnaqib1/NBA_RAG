"""Loads and validates the gold set (eval_questions.yaml) for the evaluation harness. No API calls.

Page numbers are Chroma `page` values (1-based PDF page index), the same numbering as SQLite's source_page.
See EVAL_DESIGN.md for the item schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TYPES: tuple[str, ...] = ("narrative", "numeric", "mixed", "unanswerable")
RETRIEVAL_TYPES: tuple[str, ...] = ("narrative", "mixed")
DEFAULT_GOLD = Path("eval_questions.yaml")


@dataclass(frozen=True)
class GoldItem:
    """One gold question. `retrieval_pages` are the pages search_guide should find (text pages only)."""

    id: str
    type: str
    question: str
    gold_answer: str | None
    expected_pages: tuple[int, ...]
    retrieval_pages: tuple[int, ...]
    gold_sql: str | None
    verified: bool


def _pages(value: Any, where: str, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(isinstance(p, int) and not isinstance(p, bool) and p > 0 for p in value):
        raise ValueError(f"{where}: {field} must be a list of positive page numbers")
    return tuple(value)


def _item(raw: dict[str, Any], where: str) -> GoldItem:
    qid = str(raw.get("id", "")).strip()
    kind = raw.get("type")
    if kind not in TYPES:
        raise ValueError(f"{where}: type must be one of {TYPES}")
    question = str(raw.get("question", "")).strip()
    if not question:
        raise ValueError(f"{where}: empty question")

    answer = raw.get("gold_answer")
    if kind == "unanswerable":
        if answer is not None:
            raise ValueError(f"{where}: unanswerable items must have gold_answer: null")
    elif not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"{where}: gold_answer must be a non-empty string")

    expected = _pages(raw.get("expected_pages"), where, "expected_pages")
    if kind == "unanswerable" and expected:
        raise ValueError(f"{where}: unanswerable items must have expected_pages: []")
    if kind != "unanswerable" and not expected:
        raise ValueError(f"{where}: expected_pages is empty")
    retrieval = _pages(raw["retrieval_pages"], where, "retrieval_pages") if "retrieval_pages" in raw else expected
    if not set(retrieval) <= set(expected):
        raise ValueError(f"{where}: retrieval_pages must be a subset of expected_pages")

    sql = raw.get("gold_sql")
    if kind in ("numeric", "mixed"):
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError(f"{where}: {kind} items need gold_sql")
    elif sql is not None:
        raise ValueError(f"{where}: only numeric and mixed items may have gold_sql")

    verified = raw.get("verified", False)
    if not isinstance(verified, bool):
        raise TypeError(f"{where}: verified must be true or false")
    return GoldItem(qid, kind, question, answer, expected, retrieval, sql, verified)


def load_gold(path: str | Path = DEFAULT_GOLD) -> list[GoldItem]:
    """Reads and validates every item.

    Raises:
        OSError: If the file cannot be read.
        ValueError: On a missing or duplicate id, unknown type, or a field that breaks the schema.
        TypeError: If 'questions' is not a list, an item is not a mapping, or `verified` is not a boolean.
    """
    with open(path, encoding="utf-8") as f:
        raw = (yaml.safe_load(f) or {}).get("questions") or []
    if not isinstance(raw, list):
        raise TypeError("'questions' must be a list")

    items: list[GoldItem] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw, start=1):
        qid = str(entry.get("id", "")).strip() if isinstance(entry, dict) else ""
        where = f"item #{i} ({qid or 'no id'})"
        if not isinstance(entry, dict):
            raise TypeError(f"{where}: must be a mapping")
        if not qid or qid in seen:
            raise ValueError(f"{where}: missing or duplicate id")
        seen.add(qid)
        items.append(_item(entry, where))
    return items


def select(
    items: list[GoldItem],
    types: tuple[str, ...] | None = None,
    include_unverified: bool = False,
) -> list[GoldItem]:
    """The items of the given types (all types when None), verified ones only unless `include_unverified`."""
    return [i for i in items if (types is None or i.type in types) and (include_unverified or i.verified)]
