import asyncio
import importlib.util
import logging
import math
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Coroutine, TypeVar

from sqlalchemy import update

from app.config import settings
from app.database import async_session
from app.models.chunk import DocumentChunk
from app.models.document import Document, DocumentVersion
from app.models.paper import Paper
from app.services.chunker import chunk_text, count_tokens
from app.services.embedding import embed_texts
from app.services.file_parser import parse_file
from app.services.paper_chunker import chunk_paper
from app.services.paper_parser import parse_paper_local, paper_references_to_text
from app.services.metadata_enhancer import enhance_via_crossref, enhance_via_pubmed, extract_medical_entities
from app.services.weaviate_client import COLLECTION_NAME, get_client
from app.workers.celery_app import celery_app
from weaviate.classes.query import Filter

logger = logging.getLogger(__name__)

AsyncResultT = TypeVar("AsyncResultT")
_worker_async_context = threading.local()


def _get_worker_thread_loop() -> asyncio.AbstractEventLoop:
    loop = getattr(_worker_async_context, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _worker_async_context.loop = loop
    return loop


def _run_async(coro: Coroutine[Any, Any, AsyncResultT]) -> AsyncResultT:
    """Run async work on the current Celery execution thread's event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = _get_worker_thread_loop()
        return loop.run_until_complete(coro)

    coro.close()
    raise RuntimeError("_run_async must be called from a synchronous Celery task context")


def _is_weaviate_duplicate_error(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 422 and "already exists" in str(exc)


def _upsert_weaviate_object(
    collection: Any,
    *,
    object_id: str,
    properties: dict[str, Any],
    vector: list[float],
    task_id: str,
    owner_type: str,
    owner_id: str,
) -> str:
    """Insert a Weaviate object, or update it when a retry already wrote it."""
    if collection.data.exists(object_id) is True:
        collection.data.update(uuid=object_id, properties=properties, vector=vector)
        logger.debug(
            "Weaviate object updated task_id=%s %s=%s object_id=%s",
            task_id,
            owner_type,
            owner_id,
            object_id,
        )
        return "updated"

    try:
        collection.data.insert(uuid=object_id, properties=properties, vector=vector)
        return "inserted"
    except Exception as exc:
        if not _is_weaviate_duplicate_error(exc):
            raise
        collection.data.update(uuid=object_id, properties=properties, vector=vector)
        logger.info(
            "Weaviate object duplicate on insert; updated existing object task_id=%s %s=%s object_id=%s",
            task_id,
            owner_type,
            owner_id,
            object_id,
        )
        return "updated"


def _delete_weaviate_document_chunks(collection: Any, *, org_id: str, document_id: str, version_id: str | None = None) -> int:
    conditions = [
        Filter.by_property("org_id").equal(str(org_id)),
        Filter.by_property("document_id").equal(str(document_id)),
    ]
    if version_id:
        conditions.append(Filter.by_property("document_version_id").equal(str(version_id)))

    result = collection.data.delete_many(where=Filter.all_of(conditions), verbose=True)
    matches = getattr(result, "matches", None)
    if matches is not None:
        return int(matches)
    successful = getattr(result, "successful", None)
    if successful is not None:
        return int(successful)
    objects = getattr(result, "objects", None)
    return len(objects or [])


def _make_chunk_ids(org_id: str, version_id: str, chunks: list[dict], *, namespace: str = "") -> list[str]:
    prefix = f"{org_id}:{version_id}"
    if namespace:
        prefix = f"{prefix}:{namespace}"
    return [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{prefix}:{idx}")) for idx in range(len(chunks))]


def _normalize_parent_child_metadata(chunks: list[dict], *, parent_seed: str, child_ids: list[str]) -> list[dict]:
    parent_chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, parent_seed))
    for idx, chunk in enumerate(chunks):
        chunk["chunk_index"] = idx
        chunk["parent_chunk_id"] = parent_chunk_id
        chunk["child_chunk_ids"] = child_ids
    return chunks


def _uuid_or_none(value: Any):
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except ValueError:
        return None


def _batch_publish_weaviate_chunks(collection: Any, chunk_ids: list[str], *, batch_size: int = 100) -> int:
    updated_count = 0

    for offset in range(0, len(chunk_ids), batch_size):
        batch_ids = chunk_ids[offset:offset + batch_size]
        response = collection.query.fetch_objects_by_ids(
            batch_ids,
            include_vector=True,
            return_properties=True,
        )
        objects = list(getattr(response, "objects", []) or [])
        if not objects:
            raise RuntimeError(f"Weaviate batch publish found no chunks for batch starting at offset {offset}")

        objects_by_id = {str(obj.uuid): obj for obj in objects}
        missing_ids = [cid for cid in batch_ids if cid not in objects_by_id]
        if missing_ids:
            raise RuntimeError(f"Weaviate batch publish missing {len(missing_ids)} chunks")

        with collection.batch.fixed_size(batch_size=len(batch_ids), concurrent_requests=2) as batch:
            for cid in batch_ids:
                obj = objects_by_id[cid]
                properties = dict(getattr(obj, "properties", {}) or {})
                properties["status"] = "ready"
                batch.add_object(
                    uuid=str(obj.uuid),
                    properties=properties,
                    vector=getattr(obj, "vector", None) or None,
                )
                updated_count += 1

        failed_objects = list(getattr(collection.batch, "failed_objects", []) or [])
        if failed_objects:
            raise RuntimeError(f"Weaviate batch publish failed for {len(failed_objects)} chunks")

    return updated_count


def _make_idEMPOTENCY_KEY(org_id: str, document_id: str, version_id: str, job_type: str) -> str:
    return f"{org_id}:{document_id}:{version_id}:{job_type}"


def _task_request_id(task: Any) -> str:
    return str(getattr(getattr(task, "request", None), "id", None) or "")


def _task_retries(task: Any) -> int:
    return int(getattr(getattr(task, "request", None), "retries", 0) or 0)


_EVAL_TOKEN_RE = re.compile(r"[\w\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


def _normalize_eval_config(config: dict | None) -> dict:
    config = dict(config or {})
    return {
        "top_k": int(config.get("top_k") or 10),
        "expand_query": bool(config.get("expand_query", True)),
        "generate_answers": bool(config.get("generate_answers", False)),
        "use_ragas": bool(config.get("use_ragas", False)),
        "feedback_learning": bool(config.get("feedback_learning", True)),
    }


def _rank_metrics(sources: list[Any], expected_doc_ids: list[str], *, k: int) -> dict:
    expected = {str(item) for item in expected_doc_ids or [] if item}
    if not expected:
        return {"doc_hit": False, "first_hit_rank": 0, "mrr": 0.0, "ndcg": 0.0}

    first_hit_rank = 0
    dcg = 0.0
    for rank, source in enumerate(sources[:k], start=1):
        if str(getattr(source, "document_id", "")) in expected:
            if first_hit_rank == 0:
                first_hit_rank = rank
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(expected), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        "doc_hit": first_hit_rank > 0,
        "first_hit_rank": first_hit_rank,
        "mrr": 1.0 / first_hit_rank if first_hit_rank else 0.0,
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def _kb_hit(sources: list[Any], expected_kb_ids: list[str]) -> bool:
    expected = {str(item) for item in expected_kb_ids or [] if item}
    if not expected:
        return False
    return any(str(getattr(source, "kb_id", "")) in expected for source in sources)


def _collect_generated_answer(
    *,
    query: str,
    org_id: str,
    kb_ids: list[str],
    max_chunks: int,
) -> tuple[str, list[dict]]:
    from app.services.rag import assemble_context_and_generate

    answer = ""
    sources: list[dict] = []
    for item in assemble_context_and_generate(
        query=query,
        org_id=org_id,
        kb_ids=kb_ids,
        max_chunks=max_chunks,
    ):
        answer += item.get("delta", "")
        if item.get("done"):
            sources = item.get("sources", [])
    return answer, sources


def _local_answer_quality(answer: str, expected_answer: str | None, contexts: list[str]) -> dict:
    if not expected_answer:
        return {}
    answer_terms = _eval_terms(answer)
    expected_terms = _eval_terms(expected_answer)
    context_terms = _eval_terms("\n".join(contexts))
    if not answer_terms or not expected_terms:
        relevancy = 0.0
    else:
        relevancy = len(answer_terms & expected_terms) / len(expected_terms)
    if not answer_terms or not context_terms:
        faithfulness = 0.0
    else:
        faithfulness = len(answer_terms & context_terms) / len(answer_terms)
    return {
        "answer_relevancy": round(relevancy, 4),
        "faithfulness": round(faithfulness, 4),
    }


def _eval_terms(text: str) -> set[str]:
    return {term.lower() for term in _EVAL_TOKEN_RE.findall(text or "") if len(term.strip()) >= 2}


def _ragas_available() -> bool:
    return importlib.util.find_spec("ragas") is not None


def _empty_ragas_metrics(*, requested: bool) -> dict:
    return {
        "ragas_requested": requested,
        "ragas_available": _ragas_available(),
        "ragas_evaluated": 0,
    }


def _run_ragas_metrics(records: list[dict], *, requested: bool) -> dict:
    metrics = _empty_ragas_metrics(requested=requested)
    if not requested or not records or not metrics["ragas_available"]:
        return metrics

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness

        dataset = Dataset.from_list([
            {
                "question": record["question"],
                "answer": record["answer"],
                "contexts": record["contexts"],
                "ground_truth": record["ground_truth"],
                "reference": record["ground_truth"],
            }
            for record in records
        ])
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        scores = _ragas_scores_to_dict(result)
        metrics.update({
            "ragas_evaluated": len(records),
            "ragas_faithfulness": round(float(scores.get("faithfulness", 0.0) or 0.0), 4),
            "ragas_answer_relevancy": round(float(scores.get("answer_relevancy", 0.0) or 0.0), 4),
        })
    except Exception as exc:
        metrics["ragas_error"] = str(exc)[:500]
        logger.warning("RAGAS evaluation failed: %s", exc, exc_info=True)
    return metrics


def _ragas_scores_to_dict(result: Any) -> dict:
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        return frame.mean(numeric_only=True).to_dict()
    if isinstance(result, dict):
        return result
    try:
        return dict(result)
    except (TypeError, ValueError):
        return {}


async def _mark_ingestion_job(
    document_id: str,
    version_id: str,
    status: str,
    *,
    error_code: str | None = None,
    error_message: str | None = None,
    increment_retry: bool = False,
) -> None:
    values: dict[str, Any] = {"status": status}
    now = datetime.now(timezone.utc)
    if status == "running":
        values["started_at"] = now
        values["error_code"] = None
        values["error_message"] = None
    if status in {"completed", "failed"}:
        values["finished_at"] = now
    if error_code is not None:
        values["error_code"] = error_code[:100]
    if error_message is not None:
        values["error_message"] = error_message[:2000]
    if increment_retry:
        from app.models.task import IngestionJob

        async with async_session() as session:
            await session.execute(
                update(IngestionJob)
                .where(
                    IngestionJob.document_id == document_id,
                    IngestionJob.version_id == version_id,
                    IngestionJob.job_type == "parse",
                )
                .values(**values, retry_count=IngestionJob.retry_count + 1)
            )
            await session.commit()
        return

    from app.models.task import IngestionJob

    async with async_session() as session:
        await session.execute(
            update(IngestionJob)
            .where(
                IngestionJob.document_id == document_id,
                IngestionJob.version_id == version_id,
                IngestionJob.job_type == "parse",
            )
            .values(**values)
        )
        await session.commit()


def _set_ingestion_job_status(
    document_id: str,
    version_id: str,
    status: str,
    *,
    error_code: str | None = None,
    error_message: str | None = None,
    increment_retry: bool = False,
) -> None:
    try:
        _run_async(
            _mark_ingestion_job(
                document_id=document_id,
                version_id=version_id,
                status=status,
                error_code=error_code,
                error_message=error_message,
                increment_retry=increment_retry,
            )
        )
    except Exception:
        logger.warning(
            "Failed to update ingestion job status document_id=%s version_id=%s status=%s",
            document_id,
            version_id,
            status,
            exc_info=True,
        )


def queue_document_ingestion(
    *,
    org_id: str,
    document_id: str,
    version_id: str,
    kb_id: str,
    title: str,
    storage_path: str,
):
    """Queue parse -> chunk/embed -> publish for a document."""
    ingestion_chain = (
        parse_document_task.s(
            org_id=str(org_id),
            document_id=str(document_id),
            version_id=str(version_id),
            storage_path=storage_path,
        )
        | chunk_and_embed_from_parse_task.s(
            kb_id=str(kb_id),
            title=title,
        )
        | publish_document_from_chunks_task.s()
    )
    async_result = ingestion_chain.apply_async()
    logger.info(
        "Document ingestion chain queued root_task_id=%s org_id=%s document_id=%s version_id=%s kb_id=%s",
        async_result.id,
        org_id,
        document_id,
        version_id,
        kb_id,
    )
    return async_result


def queue_document_vector_cleanup(*, org_id: str, document_id: str, version_id: str | None = None):
    """Queue asynchronous cleanup for a document's vector and chunk metadata."""
    async_result = cleanup_document_vectors_task.apply_async(
        kwargs={
            "org_id": str(org_id),
            "document_id": str(document_id),
            "version_id": str(version_id) if version_id else None,
        }
    )
    logger.info(
        "Document vector cleanup queued task_id=%s org_id=%s document_id=%s version_id=%s",
        async_result.id,
        org_id,
        document_id,
        version_id,
    )
    return async_result


async def _replace_document_chunks(
    *,
    org_id: str,
    kb_id: str,
    document_id: str,
    version_id: str,
    chunks: list[dict],
    chunk_ids: list[str],
) -> None:
    from sqlalchemy import delete

    org_uuid = uuid.UUID(str(org_id))
    kb_uuid = uuid.UUID(str(kb_id))
    document_uuid = uuid.UUID(str(document_id))
    version_uuid = uuid.UUID(str(version_id))

    async with async_session() as session:
        await session.execute(
            delete(DocumentChunk).where(
                DocumentChunk.document_id == document_uuid,
                DocumentChunk.document_version_id == version_uuid,
            )
        )
        for idx, (chunk_data, chunk_id) in enumerate(zip(chunks, chunk_ids)):
            content = chunk_data.get("content", "")
            session.add(
                DocumentChunk(
                    id=uuid.UUID(str(chunk_id)),
                    org_id=org_uuid,
                    kb_id=kb_uuid,
                    document_id=document_uuid,
                    document_version_id=version_uuid,
                    chunk_index=idx,
                    parent_chunk_id=_uuid_or_none(chunk_data.get("parent_chunk_id")),
                    weaviate_id=chunk_id,
                    content_preview=content[:300],
                    token_count=chunk_data.get("token_count"),
                    page_start=chunk_data.get("page_start"),
                    page_end=chunk_data.get("page_end"),
                    section_path=chunk_data.get("section_path"),
                    source_locator=chunk_data.get("source_locator") or {},
                    acl_hash=chunk_data.get("acl_hash"),
                )
            )
        await session.commit()


@celery_app.task(bind=True, name="parse_document", max_retries=3)
def parse_document_task(
    self,
    org_id: str,
    document_id: str,
    version_id: str,
    storage_path: str,
) -> dict:
    """Parse uploaded file and extract text."""
    import os

    try:
        _set_ingestion_job_status(document_id, version_id, "running")
        logger.info(
            "Task parse_document start task_id=%s org_id=%s document_id=%s version_id=%s storage_path=%s",
            _task_request_id(self),
            org_id,
            document_id,
            version_id,
            storage_path,
        )
        text = parse_file(storage_path)
        parsed_filename = f"{document_id}_parsed.txt"
        parsed_path = os.path.join(settings.storage_path, "parsed", parsed_filename)
        os.makedirs(os.path.dirname(parsed_path), exist_ok=True)
        with open(parsed_path, "w", encoding="utf-8") as f:
            f.write(text)

        logger.info(
            "Task parse_document complete task_id=%s org_id=%s document_id=%s version_id=%s parsed_path=%s text_length=%s",
            _task_request_id(self),
            org_id,
            document_id,
            version_id,
            parsed_path,
            len(text),
        )
        return {
            "org_id": org_id,
            "document_id": document_id,
            "version_id": version_id,
            "parsed_path": parsed_path,
            "text_length": len(text),
        }

    except Exception as e:
        _set_ingestion_job_status(
            document_id,
            version_id,
            "failed",
            error_code=e.__class__.__name__,
            error_message=str(e),
            increment_retry=True,
        )
        logger.exception(
            "Task parse_document failed task_id=%s org_id=%s document_id=%s version_id=%s retry=%s",
            _task_request_id(self),
            org_id,
            document_id,
            version_id,
            _task_retries(self),
        )
        raise self.retry(exc=e, countdown=30)


@celery_app.task(bind=True, name="chunk_and_embed", max_retries=3)
def chunk_and_embed_task(
    self,
    org_id: str,
    document_id: str,
    version_id: str,
    kb_id: str,
    title: str,
    parsed_path: str,
    batch_size: int = 10,
) -> dict:
    """Chunk parsed text, generate embeddings, and write to Weaviate."""
    try:
        logger.info(
            "Task chunk_and_embed start task_id=%s org_id=%s document_id=%s version_id=%s kb_id=%s parsed_path=%s batch_size=%s",
            _task_request_id(self),
            org_id,
            document_id,
            version_id,
            kb_id,
            parsed_path,
            batch_size,
        )
        with open(parsed_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(
            "Task chunk_and_embed parsed text loaded task_id=%s document_id=%s text_length=%s",
            _task_request_id(self),
            document_id,
            len(text),
        )

        chunks = chunk_text(text, title=title)
        chunk_ids = _make_chunk_ids(org_id, version_id, chunks)
        chunks = _normalize_parent_child_metadata(
            chunks,
            parent_seed=f"{org_id}:{version_id}:document",
            child_ids=chunk_ids,
        )
        logger.info(
            "Task chunk_and_embed chunking complete task_id=%s document_id=%s chunk_count=%s",
            _task_request_id(self),
            document_id,
            len(chunks),
        )

        # Batch embed
        all_vectors = []
        for i in range(0, len(chunks), batch_size):
            batch = [c["content"] for c in chunks[i:i + batch_size]]
            logger.info(
                "Task chunk_and_embed embedding batch task_id=%s document_id=%s batch_start=%s batch_size=%s",
                _task_request_id(self),
                document_id,
                i,
                len(batch),
            )
            vectors = embed_texts(batch)
            all_vectors.extend(vectors)
        logger.info(
            "Task chunk_and_embed embedding complete task_id=%s document_id=%s vector_count=%s",
            _task_request_id(self),
            document_id,
            len(all_vectors),
        )

        # Write to Weaviate
        client = get_client()
        logger.info("Task chunk_and_embed using Weaviate task_id=%s document_id=%s", _task_request_id(self), document_id)
        collection = client.collections.get(COLLECTION_NAME)
        deleted_count = _delete_weaviate_document_chunks(
            collection,
            org_id=org_id,
            document_id=document_id,
            version_id=version_id,
        )
        logger.info(
            "Task chunk_and_embed stale Weaviate chunks deleted task_id=%s document_id=%s version_id=%s deleted=%s",
            _task_request_id(self),
            document_id,
            version_id,
            deleted_count,
        )
        inserted_count = 0
        updated_count = 0

        for idx, (chunk_data, vector) in enumerate(zip(chunks, all_vectors)):
            weaviate_uuid = chunk_ids[idx]
            chunk_data["token_count"] = count_tokens(chunk_data.get("content", ""))
            properties = {
                "org_id": org_id,
                "kb_id": kb_id,
                "document_id": document_id,
                "document_version_id": version_id,
                "chunk_id": weaviate_uuid,
                "chunk_index": chunk_data.get("chunk_index", idx),
                "parent_chunk_id": chunk_data.get("parent_chunk_id", ""),
                "child_chunk_ids": chunk_data.get("child_chunk_ids", []),
                "security_level": "internal",
                "status": "draft",
                "content": chunk_data["content"],
                "title": title,
                "section_path": chunk_data["section_path"],
                "page_start": None,
                "page_end": None,
                "document_type": "general",
                "domain_tags": [],
                "entities": [],
                "embedding_model": settings.embedding_model,
                "created_at": datetime.now(timezone.utc),
                "section_type": None,
            }
            write_action = _upsert_weaviate_object(
                collection,
                object_id=weaviate_uuid,
                properties=properties,
                vector=vector,
                task_id=_task_request_id(self),
                owner_type="document_id",
                owner_id=document_id,
            )
            if write_action == "inserted":
                inserted_count += 1
            else:
                updated_count += 1

        try:
            _run_async(
                _replace_document_chunks(
                    org_id=org_id,
                    kb_id=kb_id,
                    document_id=document_id,
                    version_id=version_id,
                    chunks=chunks,
                    chunk_ids=chunk_ids,
                )
            )
            logger.info(
                "Task chunk_and_embed DocumentChunk rows replaced task_id=%s document_id=%s chunk_count=%s",
                _task_request_id(self),
                document_id,
                len(chunk_ids),
            )
        except ValueError:
            logger.warning(
                "Task chunk_and_embed skipped DocumentChunk persistence due to non-UUID identifiers "
                "task_id=%s org_id=%s kb_id=%s document_id=%s version_id=%s",
                _task_request_id(self),
                org_id,
                kb_id,
                document_id,
                version_id,
                exc_info=True,
            )
        except Exception:
            logger.exception(
                "Task chunk_and_embed DocumentChunk persistence failed task_id=%s document_id=%s",
                _task_request_id(self),
                document_id,
            )
            raise

        logger.info(
            "Task chunk_and_embed Weaviate upsert complete task_id=%s document_id=%s chunk_count=%s inserted=%s updated=%s",
            _task_request_id(self),
            document_id,
            len(chunk_ids),
            inserted_count,
            updated_count,
        )
        return {
            "org_id": org_id,
            "document_id": document_id,
            "version_id": version_id,
            "kb_id": kb_id,
            "chunk_count": len(chunks),
            "chunk_ids": chunk_ids,
        }

    except Exception as e:
        _set_ingestion_job_status(
            document_id,
            version_id,
            "failed",
            error_code=e.__class__.__name__,
            error_message=str(e),
            increment_retry=True,
        )
        logger.exception(
            "Task chunk_and_embed failed task_id=%s org_id=%s document_id=%s version_id=%s retry=%s",
            _task_request_id(self),
            org_id,
            document_id,
            version_id,
            _task_retries(self),
        )
        raise self.retry(exc=e, countdown=60)


@celery_app.task(bind=True, name="cleanup_document_vectors", max_retries=3)
def cleanup_document_vectors_task(
    self,
    org_id: str,
    document_id: str,
    version_id: str | None = None,
) -> dict:
    """Delete a document's vector objects and PG chunk metadata."""
    try:
        logger.info(
            "Task cleanup_document_vectors start task_id=%s org_id=%s document_id=%s version_id=%s",
            _task_request_id(self),
            org_id,
            document_id,
            version_id,
        )
        client = get_client()
        collection = client.collections.get(COLLECTION_NAME)
        deleted_vectors = _delete_weaviate_document_chunks(
            collection,
            org_id=org_id,
            document_id=document_id,
            version_id=version_id,
        )

        async def _delete_chunks() -> int:
            from sqlalchemy import delete

            try:
                document_uuid = uuid.UUID(str(document_id))
                version_uuid = uuid.UUID(str(version_id)) if version_id else None
            except ValueError:
                return 0

            async with async_session() as session:
                stmt = delete(DocumentChunk).where(DocumentChunk.document_id == document_uuid)
                if version_uuid:
                    stmt = stmt.where(DocumentChunk.document_version_id == version_uuid)
                result = await session.execute(stmt)
                await session.commit()
                return int(getattr(result, "rowcount", 0) or 0)

        deleted_chunks = _run_async(_delete_chunks())
        logger.info(
            "Task cleanup_document_vectors complete task_id=%s org_id=%s document_id=%s version_id=%s "
            "deleted_vectors=%s deleted_chunks=%s",
            _task_request_id(self),
            org_id,
            document_id,
            version_id,
            deleted_vectors,
            deleted_chunks,
        )
        return {
            "org_id": org_id,
            "document_id": document_id,
            "version_id": version_id,
            "deleted_vectors": deleted_vectors,
            "deleted_chunks": deleted_chunks,
        }
    except Exception as exc:
        logger.exception(
            "Task cleanup_document_vectors failed task_id=%s org_id=%s document_id=%s version_id=%s retry=%s",
            _task_request_id(self),
            org_id,
            document_id,
            version_id,
            _task_retries(self),
        )
        raise self.retry(exc=exc, countdown=30)


@celery_app.task(bind=True, name="chunk_and_embed_from_parse", max_retries=3)
def chunk_and_embed_from_parse_task(
    self,
    parse_result: dict,
    kb_id: str,
    title: str,
    batch_size: int = 10,
) -> dict:
    """Continue a document ingestion chain after parsing."""
    try:
        logger.info(
            "Task chunk_and_embed_from_parse start task_id=%s document_id=%s kb_id=%s text_length=%s",
            _task_request_id(self),
            parse_result.get("document_id"),
            kb_id,
            parse_result.get("text_length"),
        )
        return chunk_and_embed_task(
            parse_result["org_id"],
            parse_result["document_id"],
            parse_result["version_id"],
            kb_id,
            title,
            parse_result["parsed_path"],
            batch_size,
        )
    except Exception as e:
        logger.exception(
            "Task chunk_and_embed_from_parse failed task_id=%s document_id=%s retry=%s",
            _task_request_id(self),
            parse_result.get("document_id"),
            _task_retries(self),
        )
        raise self.retry(exc=e, countdown=60)


@celery_app.task(bind=True, name="publish_document", max_retries=1)
def publish_document_task(
    self,
    org_id: str,
    document_id: str,
    version_id: str,
    kb_id: str,
    chunk_count: int,
    chunk_ids: list[str],
) -> dict:
    """Switch document from draft to ready status."""
    try:
        logger.info(
            "Task publish_document start task_id=%s org_id=%s document_id=%s version_id=%s kb_id=%s chunk_count=%s",
            _task_request_id(self),
            org_id,
            document_id,
            version_id,
            kb_id,
            chunk_count,
        )
        async def _update():
            async with async_session() as session:
                # Update document status
                await session.execute(
                    update(Document)
                    .where(Document.id == document_id)
                    .values(status="ready")
                )
                # Update version status
                await session.execute(
                    update(DocumentVersion)
                    .where(DocumentVersion.id == version_id)
                    .values(
                        index_status="ready",
                        chunk_count=chunk_count,
                    )
                )
                # Update chunk status to ready
                for cid in chunk_ids:
                    await session.execute(
                        update(DocumentChunk)
                        .where(DocumentChunk.id == cid)
                        .values(weaviate_id=cid)
                    )
                await session.commit()

        _run_async(_update())
        logger.info(
            "Task publish_document database update complete task_id=%s document_id=%s version_id=%s",
            _task_request_id(self),
            document_id,
            version_id,
        )

        # Update Weaviate chunks status to ready
        client = get_client()
        logger.info("Task publish_document using Weaviate task_id=%s document_id=%s", _task_request_id(self), document_id)
        collection = client.collections.get(COLLECTION_NAME)
        updated_count = _batch_publish_weaviate_chunks(collection, chunk_ids)
        logger.info(
            "Task publish_document Weaviate batch update complete task_id=%s document_id=%s updated_chunks=%s requested_chunks=%s",
            _task_request_id(self),
            document_id,
            updated_count,
            len(chunk_ids),
        )

        _set_ingestion_job_status(document_id, version_id, "completed")
        logger.info("Task publish_document complete task_id=%s document_id=%s status=ready", _task_request_id(self), document_id)
        return {"document_id": document_id, "status": "ready", "chunk_count": chunk_count}

    except Exception as e:
        _set_ingestion_job_status(
            document_id,
            version_id,
            "failed",
            error_code=e.__class__.__name__,
            error_message=str(e),
            increment_retry=True,
        )
        logger.exception(
            "Task publish_document failed task_id=%s org_id=%s document_id=%s version_id=%s retry=%s",
            _task_request_id(self),
            org_id,
            document_id,
            version_id,
            _task_retries(self),
        )
        raise self.retry(exc=e, countdown=30)


@celery_app.task(bind=True, name="publish_document_from_chunks", max_retries=1)
def publish_document_from_chunks_task(self, embed_result: dict) -> dict:
    """Publish a document after chunking and embedding has completed."""
    try:
        logger.info(
            "Task publish_document_from_chunks start task_id=%s document_id=%s chunk_count=%s",
            _task_request_id(self),
            embed_result.get("document_id"),
            embed_result.get("chunk_count"),
        )
        return publish_document_task(
            embed_result["org_id"],
            embed_result["document_id"],
            embed_result["version_id"],
            embed_result["kb_id"],
            embed_result["chunk_count"],
            embed_result["chunk_ids"],
        )
    except Exception as e:
        logger.exception(
            "Task publish_document_from_chunks failed task_id=%s document_id=%s retry=%s",
            _task_request_id(self),
            embed_result.get("document_id"),
            _task_retries(self),
        )
        raise self.retry(exc=e, countdown=30)


@celery_app.task(bind=True, name="parse_paper", max_retries=3)
def parse_paper_task(
    self,
    org_id: str,
    document_id: str,
    version_id: str,
    paper_id: str,
    storage_path: str,
    doi: str | None = None,
    pmid: str | None = None,
    kb_id: str | None = None,
) -> dict:
    """Parse a SCI PDF paper, enhance metadata, chunk, and embed."""
    try:
        logger.info(
            "Task parse_paper start task_id=%s org_id=%s paper_id=%s document_id=%s version_id=%s kb_id=%s storage_path=%s doi=%s pmid=%s",
            self.request.id,
            org_id,
            paper_id,
            document_id,
            version_id,
            kb_id,
            storage_path,
            doi,
            pmid,
        )
        # Step 1: Parse the PDF
        parse_result = parse_paper_local(storage_path)
        logger.info(
            "Task parse_paper parse complete task_id=%s paper_id=%s title_present=%s abstract_length=%s section_count=%s parser=%s",
            self.request.id,
            paper_id,
            bool(parse_result.title),
            len(parse_result.abstract or ""),
            len(parse_result.sections),
            parse_result.parser,
        )
        if doi:
            crossref_meta = enhance_via_crossref(doi)
            if crossref_meta:
                if not parse_result.title and crossref_meta.title:
                    parse_result.title = crossref_meta.title
                if crossref_meta.abstract and not parse_result.abstract:
                    parse_result.abstract = crossref_meta.abstract
                if crossref_meta.journal:
                    parse_result.journal = crossref_meta.journal
                logger.info(
                    "Task parse_paper CrossRef enhancement applied task_id=%s paper_id=%s doi=%s has_title=%s has_abstract=%s",
                    self.request.id,
                    paper_id,
                    doi,
                    bool(crossref_meta.title),
                    bool(crossref_meta.abstract),
                )
            else:
                logger.warning("Task parse_paper CrossRef enhancement returned no metadata task_id=%s paper_id=%s doi=%s", self.request.id, paper_id, doi)
        if pmid:
            pubmed_meta = enhance_via_pubmed(pmid)
            if pubmed_meta:
                if not parse_result.title and pubmed_meta.title:
                    parse_result.title = pubmed_meta.title
                logger.info(
                    "Task parse_paper PubMed enhancement applied task_id=%s paper_id=%s pmid=%s has_title=%s mesh_count=%s",
                    self.request.id,
                    paper_id,
                    pmid,
                    bool(pubmed_meta.title),
                    len(pubmed_meta.mesh_terms),
                )
            else:
                logger.warning("Task parse_paper PubMed enhancement returned no metadata task_id=%s paper_id=%s pmid=%s", self.request.id, paper_id, pmid)

        # Step 2: Extract medical entities
        full_text = (parse_result.abstract or "") + " " + " ".join(
            s.content for s in parse_result.sections
        )
        entities = extract_medical_entities(full_text)
        logger.info(
            "Task parse_paper entities extracted task_id=%s paper_id=%s entity_counts=%s",
            self.request.id,
            paper_id,
            {k: len(v) for k, v in entities.items()},
        )

        # Step 3: Update Paper record with parsed data
        resolved_kb_id = kb_id

        async def _update_paper():
            nonlocal resolved_kb_id
            async with async_session() as session:
                paper = await session.get(Paper, paper_id)
                if paper:
                    if not resolved_kb_id and paper.kb_id:
                        resolved_kb_id = str(paper.kb_id)
                    if parse_result.title:
                        paper.title = parse_result.title
                    if parse_result.abstract:
                        paper.abstract = parse_result.abstract
                    if parse_result.journal:
                        paper.journal = parse_result.journal
                    if parse_result.authors:
                        import json
                        paper.authors = json.dumps(parse_result.authors)
                    if parse_result.doi:
                        paper.doi = parse_result.doi
                    paper.mesh_terms = entities.get("diseases", [])[:20]
                    paper.diseases = entities.get("diseases", [])[:20]
                    paper.drugs = entities.get("drugs", [])[:20]
                    paper.targets = entities.get("targets", [])[:20]
                    paper.parser_version = "paper_parser_v1"
                    paper.grobid_confidence = parse_result.grobid_confidence
                    paper.enhancement_source = "crossref" if doi else ("pubmed" if pmid else "none")
                    await session.commit()

        _run_async(_update_paper())
        logger.info(
            "Task parse_paper paper record updated task_id=%s paper_id=%s resolved_kb_id=%s",
            self.request.id,
            paper_id,
            resolved_kb_id,
        )

        # Step 4: Chunk parsed paper content
        ref_text = paper_references_to_text(parse_result)
        title = parse_result.title or f"Paper {document_id}"

        # Use paper-specific chunking
        paper_chunks = chunk_paper(parse_result, title=title)

        # Also add reference text as a separate chunk if available
        if ref_text.strip():
            paper_chunks.append({
                "content": ref_text,
                "section_path": f"{title}/References",
                "section_type": "references",
                "page_start": None,
                "page_end": None,
                "boost": 0.5,
            })
        chunk_ids = _make_chunk_ids(org_id, version_id, paper_chunks, namespace="paper")
        paper_chunks = _normalize_parent_child_metadata(
            paper_chunks,
            parent_seed=f"{org_id}:{version_id}:paper",
            child_ids=chunk_ids,
        )
        logger.info(
            "Task parse_paper chunking complete task_id=%s paper_id=%s chunk_count=%s reference_text_length=%s",
            self.request.id,
            paper_id,
            len(paper_chunks),
            len(ref_text),
        )

        # Step 5: Embed chunks
        batch_size = 10
        all_vectors = []
        for i in range(0, len(paper_chunks), batch_size):
            batch = [c["content"] for c in paper_chunks[i:i + batch_size]]
            logger.info(
                "Task parse_paper embedding batch task_id=%s paper_id=%s batch_start=%s batch_size=%s",
                self.request.id,
                paper_id,
                i,
                len(batch),
            )
            vectors = embed_texts(batch)
            all_vectors.extend(vectors)
        logger.info("Task parse_paper embedding complete task_id=%s paper_id=%s vector_count=%s", self.request.id, paper_id, len(all_vectors))

        # Step 6: Write to Weaviate
        client = get_client()
        logger.info("Task parse_paper using Weaviate task_id=%s paper_id=%s", self.request.id, paper_id)
        collection = client.collections.get(COLLECTION_NAME)
        deleted_count = _delete_weaviate_document_chunks(
            collection,
            org_id=org_id,
            document_id=document_id,
            version_id=version_id,
        )
        logger.info(
            "Task parse_paper stale Weaviate chunks deleted task_id=%s paper_id=%s document_id=%s version_id=%s deleted=%s",
            self.request.id,
            paper_id,
            document_id,
            version_id,
            deleted_count,
        )
        inserted_count = 0
        updated_count = 0

        for idx, (chunk_data, vector) in enumerate(zip(paper_chunks, all_vectors)):
            weaviate_uuid = chunk_ids[idx]
            chunk_data["token_count"] = count_tokens(chunk_data.get("content", ""))
            properties = {
                "org_id": org_id,
                "kb_id": resolved_kb_id or "",
                "document_id": document_id,
                "document_version_id": version_id,
                "chunk_id": weaviate_uuid,
                "chunk_index": chunk_data.get("chunk_index", idx),
                "parent_chunk_id": chunk_data.get("parent_chunk_id", ""),
                "child_chunk_ids": chunk_data.get("child_chunk_ids", []),
                "security_level": "internal",
                "status": "ready",  # Papers go directly to ready
                "content": chunk_data["content"],
                "title": title,
                "section_path": chunk_data.get("section_path", ""),
                "page_start": chunk_data.get("page_start"),
                "page_end": chunk_data.get("page_end"),
                "document_type": "paper",
                "domain_tags": entities.get("diseases", [])[:5] + entities.get("targets", [])[:5],
                "entities": entities.get("drugs", [])[:10] + entities.get("targets", [])[:10],
                "embedding_model": settings.embedding_model,
                "created_at": datetime.now(timezone.utc),
                "section_type": chunk_data.get("section_type", "other"),
            }
            write_action = _upsert_weaviate_object(
                collection,
                object_id=weaviate_uuid,
                properties=properties,
                vector=vector,
                task_id=self.request.id,
                owner_type="paper_id",
                owner_id=paper_id,
            )
            if write_action == "inserted":
                inserted_count += 1
            else:
                updated_count += 1
        try:
            _run_async(
                _replace_document_chunks(
                    org_id=org_id,
                    kb_id=resolved_kb_id or "",
                    document_id=document_id,
                    version_id=version_id,
                    chunks=paper_chunks,
                    chunk_ids=chunk_ids,
                )
            )
            logger.info(
                "Task parse_paper DocumentChunk rows replaced task_id=%s paper_id=%s document_id=%s chunk_count=%s",
                self.request.id,
                paper_id,
                document_id,
                len(chunk_ids),
            )
        except ValueError:
            logger.warning(
                "Task parse_paper skipped DocumentChunk persistence due to non-UUID identifiers "
                "task_id=%s org_id=%s kb_id=%s document_id=%s version_id=%s",
                self.request.id,
                org_id,
                resolved_kb_id,
                document_id,
                version_id,
                exc_info=True,
            )
        except Exception:
            logger.exception(
                "Task parse_paper DocumentChunk persistence failed task_id=%s paper_id=%s document_id=%s",
                self.request.id,
                paper_id,
                document_id,
            )
            raise
        logger.info(
            "Task parse_paper Weaviate upsert complete task_id=%s paper_id=%s chunk_count=%s inserted=%s updated=%s",
            self.request.id,
            paper_id,
            len(chunk_ids),
            inserted_count,
            updated_count,
        )

        # Step 7: Update document status
        async def _finalize():
            async with async_session() as session:
                await session.execute(
                    update(Document)
                    .where(Document.id == document_id)
                    .values(status="ready", document_type="paper")
                )
                await session.execute(
                    update(DocumentVersion)
                    .where(DocumentVersion.id == version_id)
                    .values(index_status="ready", chunk_count=len(paper_chunks))
                )
                await session.execute(
                    update(Paper)
                    .where(Paper.id == paper_id)
                    .values(status="ready")
                )
                await session.commit()

        _run_async(_finalize())
        logger.info(
            "Task parse_paper finalize complete task_id=%s paper_id=%s document_id=%s chunk_count=%s",
            self.request.id,
            paper_id,
            document_id,
            len(paper_chunks),
        )

        return {
            "paper_id": paper_id,
            "document_id": document_id,
            "kb_id": resolved_kb_id or "",
            "status": "ready",
            "chunk_count": len(paper_chunks),
            "chunk_ids": chunk_ids,
        }

    except Exception as e:
        logger.exception(
            "Task parse_paper failed task_id=%s org_id=%s paper_id=%s document_id=%s version_id=%s retry=%s",
            self.request.id,
            org_id,
            paper_id,
            document_id,
            version_id,
            self.request.retries,
        )
        raise self.retry(exc=e, countdown=30)


@celery_app.task(bind=True, name="run_evaluation", max_retries=1)
def run_evaluation_task(
    self,
    org_id: str,
    run_id: str,
    eval_set_id: str,
    config: dict | None = None,
) -> dict:
    """Run evaluation against an evaluation set."""
    from sqlalchemy import select
    from app.models.audit import EvaluationQuestion, EvaluationRun
    from app.services.feedback_learning import load_feedback_weights
    from app.services.rag import retrieve_sources
    from app.services.rag_trace import trace_collector

    try:
        eval_config = _normalize_eval_config(config)
        logger.info(
            "Task run_evaluation start task_id=%s org_id=%s run_id=%s eval_set_id=%s config=%s",
            self.request.id,
            org_id,
            run_id,
            eval_set_id,
            eval_config,
        )
        async def _run():
            async with async_session() as session:
                # Update run status to running
                await session.execute(
                    update(EvaluationRun)
                    .where(EvaluationRun.id == run_id)
                    .values(status="running")
                )
                await session.commit()
                logger.info("Task run_evaluation status updated task_id=%s run_id=%s status=running", self.request.id, run_id)

                # Get questions
                q_stmt = select(EvaluationQuestion).where(
                    EvaluationQuestion.eval_set_id == eval_set_id,
                    EvaluationQuestion.org_id == org_id,
                )
                result = await session.execute(q_stmt)
                questions = list(result.scalars().all())
                logger.info(
                    "Task run_evaluation questions loaded task_id=%s run_id=%s question_count=%s",
                    self.request.id,
                    run_id,
                    len(questions),
                )

                metrics = {
                    "total_questions": len(questions),
                    "answered": 0,
                    "zero_result": 0,
                    "correct_kb_hit": 0,
                    "correct_doc_hit": 0,
                    "mrr_at_10": 0.0,
                    "ndcg_at_10": 0.0,
                    "evaluated_answers": 0,
                    "answer_relevancy": 0.0,
                    "faithfulness": 0.0,
                }
                local_answer_relevancy = []
                local_faithfulness = []
                ragas_records = []
                feedback_weights = None
                if eval_config["feedback_learning"]:
                    feedback_weights = await load_feedback_weights(session, org_id)
                    metrics["feedback_weight_samples"] = feedback_weights.sample_count

                for q in questions:
                    logger.debug(
                        "Task run_evaluation question search start task_id=%s run_id=%s question_id=%s expected_kb_count=%s expected_doc_count=%s",
                        self.request.id,
                        run_id,
                        q.id,
                        len(q.expected_kb_ids or []),
                        len(q.expected_doc_ids or []),
                    )
                    trace_id = str(uuid.uuid4())
                    trace = trace_collector.start_trace(
                        trace_id,
                        org_id,
                        "evaluation",
                        q.question,
                        q.expected_kb_ids or [],
                    )

                    sources = retrieve_sources(
                        query=q.question,
                        org_id=org_id,
                        kb_ids=q.expected_kb_ids or [],
                        top_k=eval_config["top_k"],
                        expand_query=eval_config["expand_query"],
                        trace=trace,
                        feedback_weights=feedback_weights,
                    )
                    logger.debug(
                        "Task run_evaluation question search complete task_id=%s run_id=%s question_id=%s source_count=%s",
                        self.request.id,
                        run_id,
                        q.id,
                        len(sources),
                    )

                    metrics["answered"] += 1
                    if not sources:
                        metrics["zero_result"] += 1
                        continue

                    if _kb_hit(sources, q.expected_kb_ids or []):
                        metrics["correct_kb_hit"] += 1

                    rank_metrics = _rank_metrics(sources, q.expected_doc_ids or [], k=eval_config["top_k"])
                    if rank_metrics["doc_hit"]:
                        metrics["correct_doc_hit"] += 1
                    metrics["mrr_at_10"] += rank_metrics["mrr"]
                    metrics["ndcg_at_10"] += rank_metrics["ndcg"]

                    if eval_config["generate_answers"] and q.expected_answer:
                        answer, generated_sources = _collect_generated_answer(
                            query=q.question,
                            org_id=org_id,
                            kb_ids=q.expected_kb_ids or [],
                            max_chunks=eval_config["top_k"],
                        )
                        contexts = [
                            str(getattr(source, "content", "") or "")
                            for source in sources
                        ]
                        quality = _local_answer_quality(answer, q.expected_answer, contexts)
                        if quality:
                            metrics["evaluated_answers"] += 1
                            local_answer_relevancy.append(quality["answer_relevancy"])
                            local_faithfulness.append(quality["faithfulness"])
                            ragas_records.append({
                                "question": q.question,
                                "answer": answer,
                                "contexts": contexts,
                                "ground_truth": q.expected_answer,
                                "generated_source_count": len(generated_sources),
                            })

                # Compute final metrics
                total = metrics["total_questions"]
                metrics["recall_at_10"] = metrics["correct_doc_hit"] / max(total, 1)
                metrics["zero_result_rate"] = metrics["zero_result"] / max(total, 1)
                metrics["kb_hit_rate_at_10"] = metrics["correct_kb_hit"] / max(total, 1)
                metrics["mrr_at_10"] = metrics["mrr_at_10"] / max(total, 1)
                metrics["ndcg_at_10"] = metrics["ndcg_at_10"] / max(total, 1)
                if local_answer_relevancy:
                    metrics["answer_relevancy"] = sum(local_answer_relevancy) / len(local_answer_relevancy)
                if local_faithfulness:
                    metrics["faithfulness"] = sum(local_faithfulness) / len(local_faithfulness)
                metrics.update(_run_ragas_metrics(ragas_records, requested=eval_config["use_ragas"]))

                # Update run status to completed
                await session.execute(
                    update(EvaluationRun)
                    .where(EvaluationRun.id == run_id)
                    .values(
                        status="completed",
                        metrics=metrics,
                    )
                )
                await session.commit()
                logger.info("Task run_evaluation complete task_id=%s run_id=%s metrics=%s", self.request.id, run_id, metrics)
                return metrics

        metrics = _run_async(_run())
        logger.info("Task run_evaluation finished task_id=%s run_id=%s", self.request.id, run_id)
        return metrics

    except Exception as exc:
        logger.exception(
            "Task run_evaluation failed task_id=%s org_id=%s run_id=%s eval_set_id=%s retry=%s",
            self.request.id,
            org_id,
            run_id,
            eval_set_id,
            self.request.retries,
        )
        error_message = str(exc)

        async def _fail():
            async with async_session() as session:
                await session.execute(
                    update(EvaluationRun)
                    .where(EvaluationRun.id == run_id)
                    .values(status="failed", error_message=error_message)
                )
                await session.commit()

        _run_async(_fail())
        logger.info("Task run_evaluation status updated task_id=%s run_id=%s status=failed", self.request.id, run_id)
        raise self.retry(exc=exc, countdown=60)
