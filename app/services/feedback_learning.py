"""Feedback-derived retrieval weighting.

This module turns stored answer feedback into lightweight online weights. It is
not an offline LTR trainer; it provides a deterministic bridge from user ratings
to rerank adjustments that can run with the current database schema.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from app.config import settings
from app.models.audit import AnswerFeedback
from app.models.conversation import ConversationMessage

logger = logging.getLogger(__name__)

_feedback_weight_cache: dict[tuple[str, int], tuple[float, "FeedbackWeights"]] = {}


@dataclass
class FeedbackWeights:
    chunk_weights: dict[str, float] = field(default_factory=dict)
    document_weights: dict[str, float] = field(default_factory=dict)
    sample_count: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_weights": self.chunk_weights,
            "document_weights": self.document_weights,
            "sample_count": self.sample_count,
        }


def build_feedback_weights(records: list[tuple[int, list[dict[str, Any]]]]) -> FeedbackWeights:
    chunk_totals: dict[str, float] = {}
    document_totals: dict[str, float] = {}
    sample_count = 0

    for rating, sources in records:
        weight_delta = _rating_delta(rating)
        if weight_delta == 0.0:
            continue
        sample_count += 1
        for source in sources or []:
            chunk_id = str(source.get("chunk_id") or "")
            document_id = str(source.get("document_id") or "")
            if chunk_id:
                chunk_totals[chunk_id] = chunk_totals.get(chunk_id, 0.0) + weight_delta
            if document_id:
                document_totals[document_id] = document_totals.get(document_id, 0.0) + weight_delta

    return FeedbackWeights(
        chunk_weights={key: _clamp_feedback_weight(value) for key, value in chunk_totals.items()},
        document_weights={key: _clamp_feedback_weight(value) for key, value in document_totals.items()},
        sample_count=sample_count,
    )


async def load_feedback_weights(session, org_id: str, *, limit: int | None = None) -> FeedbackWeights:
    limit = limit or int(getattr(settings, "feedback_learning_window", 200) or 200)
    cache_key = (str(org_id), limit)
    ttl = int(getattr(settings, "feedback_learning_cache_ttl", 300) or 300)
    now = time.monotonic()
    cached = _feedback_weight_cache.get(cache_key)
    if cached and cached[0] > now:
        return cached[1]

    stmt = (
        select(AnswerFeedback.rating, ConversationMessage.sources)
        .join(ConversationMessage, ConversationMessage.id == AnswerFeedback.message_id)
        .where(AnswerFeedback.org_id == str(org_id))
        .order_by(AnswerFeedback.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    records = []
    for row in result.all():
        rating = int(row[0] or 0)
        sources = row[1] or []
        records.append((rating, sources))
    weights = build_feedback_weights(records)
    _feedback_weight_cache[cache_key] = (now + ttl, weights)
    logger.info(
        "Feedback weights loaded org_id=%s sample_count=%s chunk_weights=%s document_weights=%s",
        org_id,
        weights.sample_count,
        len(weights.chunk_weights),
        len(weights.document_weights),
    )
    return weights


def clear_feedback_weight_cache(org_id: str | None = None) -> None:
    if org_id is None:
        _feedback_weight_cache.clear()
        return
    prefix = str(org_id)
    for key in list(_feedback_weight_cache):
        if key[0] == prefix:
            _feedback_weight_cache.pop(key, None)


def apply_feedback_weights(sources: list[Any], weights: FeedbackWeights, *, strength: float | None = None) -> list[Any]:
    if not sources or not weights.sample_count:
        return sources
    strength = strength if strength is not None else float(getattr(settings, "feedback_rerank_strength", 0.15) or 0.15)
    if strength <= 0:
        return sources

    for source in sources:
        delta = weights.chunk_weights.get(str(getattr(source, "chunk_id", "") or ""), 0.0)
        delta += weights.document_weights.get(str(getattr(source, "document_id", "") or ""), 0.0)
        if delta == 0.0:
            continue
        source.feedback_score = delta
        source.score = float(getattr(source, "score", 0.0) or 0.0) + strength * delta

    sources.sort(key=lambda item: float(getattr(item, "score", 0.0) or 0.0), reverse=True)
    return sources


def _rating_delta(rating: int) -> float:
    if rating >= 4:
        return 1.0
    if rating <= 2:
        return -1.0
    return 0.0


def _clamp_feedback_weight(value: float) -> float:
    return max(-3.0, min(3.0, value))
