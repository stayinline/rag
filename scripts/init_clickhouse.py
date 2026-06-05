"""Initialize ClickHouse analytics tables and optionally backfill traces."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import timezone
from typing import Any

import httpx
from sqlalchemy import text

sys.path.insert(0, ".")

from app.config import settings
from app.database import async_session


CREATE_RAG_TRACE_EVENTS = """
CREATE TABLE IF NOT EXISTS rag_trace_events (
    event_time DateTime64(3),
    trace_id String,
    org_id UUID,
    user_id UUID,
    scenario LowCardinality(String),
    query_hash String,
    query_text String,
    kb_ids Array(UUID),
    rewrite_count UInt8,
    retrieved_count UInt16,
    reranked_count UInt16,
    source_count UInt8,
    latency_ms UInt32,
    first_token_ms UInt32,
    model String,
    prompt_version String,
    input_tokens UInt32,
    output_tokens UInt32,
    cost_units Float64,
    safety_flags Array(String),
    rating Nullable(UInt8)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (org_id, event_time, scenario)
"""


CREATE_RETRIEVAL_HIT_EVENTS = """
CREATE TABLE IF NOT EXISTS retrieval_hit_events (
    event_time DateTime64(3),
    trace_id String,
    org_id UUID,
    query_hash String,
    chunk_id UUID,
    document_id UUID,
    rank_before UInt16,
    rank_after UInt16,
    vector_score Float32,
    bm25_score Float32,
    rerank_score Float32,
    clicked Bool,
    cited Bool
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (org_id, event_time, document_id)
"""


async def _clickhouse_post(query: str, data: str | None = None) -> httpx.Response:
    auth = (settings.clickhouse_user, settings.clickhouse_password) if settings.clickhouse_user else None
    async with httpx.AsyncClient(timeout=30, auth=auth) as client:
        return await client.post(
            settings.clickhouse_url,
            params={"database": settings.clickhouse_database, "query": query},
            content=data,
        )


async def _execute_clickhouse(query: str) -> None:
    resp = await _clickhouse_post(query)
    if resp.status_code != 200:
        raise RuntimeError(f"ClickHouse query failed: {resp.status_code} {resp.text[:500]}")


async def create_tables() -> None:
    await _execute_clickhouse(CREATE_RAG_TRACE_EVENTS)
    await _execute_clickhouse(CREATE_RETRIEVAL_HIT_EVENTS)


async def _existing_trace_ids() -> set[str]:
    resp = await _clickhouse_post("SELECT trace_id FROM rag_trace_events FORMAT JSON")
    if resp.status_code != 200:
        raise RuntimeError(f"Could not read existing traces: {resp.status_code} {resp.text[:500]}")
    return {row["trace_id"] for row in resp.json().get("data", [])}


def _format_datetime(value: Any) -> str:
    if value is None:
        raise ValueError("event_time is required for backfill")
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.strftime("%Y-%m-%d %H:%M:%S.%f")[:23]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _query_hash(org_id: str, query: str) -> str:
    return hashlib.md5(f"{org_id}:{query}".encode()).hexdigest()[:16]


async def backfill_from_postgres() -> int:
    existing = await _existing_trace_ids()
    async with async_session() as session:
        result = await session.execute(text("""
            SELECT
                a.trace_id,
                a.org_id::text AS org_id,
                a.user_id::text AS user_id,
                a.created_at,
                COALESCE(u.content, '') AS query_text,
                a.kb_ids,
                a.sources,
                COALESCE(a.model, '') AS model,
                COALESCE(a.prompt_version, '') AS prompt_version,
                fb.rating
            FROM conversation_messages a
            LEFT JOIN conversation_messages u
                ON u.org_id = a.org_id
                AND u.user_id = a.user_id
                AND u.conversation_id = a.conversation_id
                AND u.sequence = a.sequence - 1
                AND u.role = 'user'
            LEFT JOIN LATERAL (
                SELECT rating
                FROM answer_feedback af
                WHERE af.org_id = a.org_id
                  AND af.trace_id = a.trace_id
                ORDER BY af.created_at DESC
                LIMIT 1
            ) fb ON TRUE
            WHERE a.role = 'assistant'
              AND a.trace_id IS NOT NULL
            ORDER BY a.created_at ASC
        """))
        rows = result.mappings().all()

    payload_rows = []
    for row in rows:
        trace_id = row["trace_id"]
        if trace_id in existing:
            continue
        sources = _as_list(row["sources"])
        kb_ids = [str(kb_id) for kb_id in _as_list(row["kb_ids"]) if kb_id]
        query_text = str(row["query_text"] or "")
        source_count = min(len(sources), 255)
        payload_rows.append({
            "event_time": _format_datetime(row["created_at"]),
            "trace_id": trace_id,
            "org_id": row["org_id"],
            "user_id": row["user_id"],
            "scenario": "qa",
            "query_hash": _query_hash(row["org_id"], query_text),
            "query_text": query_text[:2000],
            "kb_ids": kb_ids,
            "rewrite_count": 0,
            "retrieved_count": source_count,
            "reranked_count": source_count,
            "source_count": source_count,
            "latency_ms": 0,
            "first_token_ms": 0,
            "model": row["model"],
            "prompt_version": row["prompt_version"],
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_units": 0.0,
            "safety_flags": [],
            "rating": row["rating"],
        })

    if not payload_rows:
        return 0

    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in payload_rows)
    resp = await _clickhouse_post("INSERT INTO rag_trace_events FORMAT JSONEachRow", payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Backfill insert failed: {resp.status_code} {resp.text[:500]}")
    return len(payload_rows)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backfill-from-postgres",
        action="store_true",
        help="Backfill rag_trace_events from stored conversation messages and feedback.",
    )
    args = parser.parse_args()

    print(f"Initializing ClickHouse database {settings.clickhouse_database!r} at {settings.clickhouse_url}...")
    await create_tables()
    print("ClickHouse analytics tables are ready.")

    if args.backfill_from_postgres:
        inserted = await backfill_from_postgres()
        print(f"Backfilled {inserted} trace event(s) from PostgreSQL.")


if __name__ == "__main__":
    asyncio.run(main())
