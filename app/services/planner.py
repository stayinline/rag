"""Deterministic query planner for decomposition and multi-hop retrieval."""
from __future__ import annotations

import logging
import re

from app.config import settings

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"[?？!！;；。]\s*")
_CLAUSE_SPLIT_RE = re.compile(
    r"\s+(?:and|then|versus|vs\.?|compare|compared with)\s+|(?:并且|同时|然后|以及|对比|比较)"
)


class QueryPlan:
    def __init__(self, original: str, queries: list[str], enabled: bool, strategy: str, reason: str):
        self.original = original
        self.queries = queries
        self.enabled = enabled
        self.strategy = strategy
        self.reason = reason

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "queries": self.queries,
            "enabled": self.enabled,
            "strategy": self.strategy,
            "reason": self.reason,
        }


def build_query_plan(query: str, *, mode: str | None = None, max_queries: int | None = None) -> QueryPlan:
    """Build a retrieval plan from a user query.

    The project has a `planner_mode=langgraph` config value but no LangGraph runtime
    dependency. This function treats that mode as an instruction to enable a
    deterministic local planner.
    """
    original = " ".join((query or "").split())
    mode = (mode if mode is not None else getattr(settings, "planner_mode", "")).lower()
    max_queries = max_queries or int(getattr(settings, "planner_max_tool_calls", 10) or 10)
    max_queries = max(1, min(max_queries, 10))

    if not original:
        return QueryPlan(original=query or "", queries=[], enabled=False, strategy="none", reason="empty_query")
    if mode in {"", "none", "off", "disabled"}:
        return QueryPlan(original=original, queries=[original], enabled=False, strategy="none", reason="disabled")

    subqueries = _decompose_query(original, max_queries=max_queries)
    if len(subqueries) <= 1:
        return QueryPlan(original=original, queries=[original], enabled=False, strategy="deterministic", reason="single_hop")

    queries = [original]
    for item in subqueries:
        if item not in queries:
            queries.append(item)
    queries = queries[:max_queries]
    logger.info(
        "Query planner generated plan mode=%s query_length=%s query_count=%s",
        mode,
        len(original),
        len(queries),
    )
    return QueryPlan(original=original, queries=queries, enabled=True, strategy="deterministic", reason="decomposed")


def _decompose_query(query: str, *, max_queries: int) -> list[str]:
    candidates: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(query):
        sentence = sentence.strip(" \t\r\n,，")
        if not sentence:
            continue
        candidates.extend(_split_clause(sentence))

    normalized = []
    seen = set()
    for item in candidates:
        item = " ".join(item.strip(" \t\r\n,，").split())
        if len(item) < 4 or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
        if len(normalized) >= max_queries:
            break
    return normalized or [query]


def _split_clause(value: str) -> list[str]:
    parts = [part.strip(" \t\r\n,，") for part in _CLAUSE_SPLIT_RE.split(value) if part.strip(" \t\r\n,，")]
    if len(parts) <= 1:
        return [value]
    long_parts = [part for part in parts if len(part) >= 6]
    return long_parts if len(long_parts) >= 2 else [value]
