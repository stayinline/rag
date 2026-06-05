import asyncio
import logging
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
from app.services.chunker import chunk_text
from app.services.embedding import embed_texts
from app.services.file_parser import parse_file
from app.services.paper_chunker import chunk_paper
from app.services.paper_parser import parse_paper_local, paper_references_to_text
from app.services.metadata_enhancer import enhance_via_crossref, enhance_via_pubmed, extract_medical_entities
from app.services.weaviate_client import COLLECTION_NAME, get_client
from app.workers.celery_app import celery_app

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
        chunk_ids = []
        inserted_count = 0
        updated_count = 0

        for idx, (chunk_data, vector) in enumerate(zip(chunks, all_vectors)):
            weaviate_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{org_id}:{version_id}:{idx}"))
            properties = {
                "org_id": org_id,
                "kb_id": kb_id,
                "document_id": document_id,
                "document_version_id": version_id,
                "chunk_id": weaviate_uuid,
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
            chunk_ids.append(weaviate_uuid)

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
        chunk_ids = []
        inserted_count = 0
        updated_count = 0

        for idx, (chunk_data, vector) in enumerate(zip(paper_chunks, all_vectors)):
            weaviate_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{org_id}:{version_id}:paper:{idx}"))
            properties = {
                "org_id": org_id,
                "kb_id": resolved_kb_id or "",
                "document_id": document_id,
                "document_version_id": version_id,
                "chunk_id": weaviate_uuid,
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
            chunk_ids.append(weaviate_uuid)
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
    from app.services.rag import hybrid_search

    try:
        logger.info(
            "Task run_evaluation start task_id=%s org_id=%s run_id=%s eval_set_id=%s",
            self.request.id,
            org_id,
            run_id,
            eval_set_id,
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
                }

                for q in questions:
                    logger.debug(
                        "Task run_evaluation question search start task_id=%s run_id=%s question_id=%s expected_kb_count=%s expected_doc_count=%s",
                        self.request.id,
                        run_id,
                        q.id,
                        len(q.expected_kb_ids or []),
                        len(q.expected_doc_ids or []),
                    )
                    # Run search
                    sources = hybrid_search(
                        query=q.question,
                        org_id=org_id,
                        kb_ids=q.expected_kb_ids or [],
                        top_k=10,
                    )
                    retrieved_doc_ids = {s.document_id for s in sources}
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

                    # Check if expected KBs are in results
                    if q.expected_kb_ids:
                        metrics["correct_kb_hit"] += 1  # simplified

                    # Check if expected docs are in results
                    if q.expected_doc_ids:
                        expected = set(q.expected_doc_ids)
                        hits = expected & retrieved_doc_ids
                        if hits:
                            metrics["correct_doc_hit"] += 1

                # Compute final metrics
                total = metrics["total_questions"]
                metrics["recall_at_10"] = metrics["correct_doc_hit"] / max(total, 1)
                metrics["zero_result_rate"] = metrics["zero_result"] / max(total, 1)

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
