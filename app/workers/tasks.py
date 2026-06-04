import uuid
from datetime import datetime, timezone

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


def _make_idEMPOTENCY_KEY(org_id: str, document_id: str, version_id: str, job_type: str) -> str:
    return f"{org_id}:{document_id}:{version_id}:{job_type}"


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
        text = parse_file(storage_path)
        parsed_filename = f"{document_id}_parsed.txt"
        parsed_path = os.path.join(settings.storage_path, "parsed", parsed_filename)
        os.makedirs(os.path.dirname(parsed_path), exist_ok=True)
        with open(parsed_path, "w", encoding="utf-8") as f:
            f.write(text)

        return {
            "org_id": org_id,
            "document_id": document_id,
            "version_id": version_id,
            "parsed_path": parsed_path,
            "text_length": len(text),
        }

    except Exception as e:
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
        with open(parsed_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text, title=title)

        # Batch embed
        all_vectors = []
        for i in range(0, len(chunks), batch_size):
            batch = [c["content"] for c in chunks[i:i + batch_size]]
            vectors = embed_texts(batch)
            all_vectors.extend(vectors)

        # Write to Weaviate
        client = get_client()
        client.connect()
        try:
            collection = client.collections.get(COLLECTION_NAME)
            chunk_ids = []

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
                collection.data.insert(
                    uuid=weaviate_uuid,
                    properties=properties,
                    vector=vector,
                )
                chunk_ids.append(weaviate_uuid)

            return {
                "org_id": org_id,
                "document_id": document_id,
                "version_id": version_id,
                "kb_id": kb_id,
                "chunk_count": len(chunks),
                "chunk_ids": chunk_ids,
            }
        finally:
            client.close()

    except Exception as e:
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

        import asyncio
        asyncio.run(_update())

        # Update Weaviate chunks status to ready
        client = get_client()
        client.connect()
        try:
            collection = client.collections.get(COLLECTION_NAME)
            for cid in chunk_ids:
                try:
                    collection.data.update(
                        uuid=cid,
                        properties={"status": "ready"},
                    )
                except Exception:
                    pass
        finally:
            client.close()

        return {"document_id": document_id, "status": "ready", "chunk_count": chunk_count}

    except Exception as e:
        raise self.retry(exc=e, countdown=30)


@celery_app.task(bind=True, name="publish_document_from_chunks", max_retries=1)
def publish_document_from_chunks_task(self, embed_result: dict) -> dict:
    """Publish a document after chunking and embedding has completed."""
    try:
        return publish_document_task(
            embed_result["org_id"],
            embed_result["document_id"],
            embed_result["version_id"],
            embed_result["kb_id"],
            embed_result["chunk_count"],
            embed_result["chunk_ids"],
        )
    except Exception as e:
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
        # Step 1: Parse the PDF
        parse_result = parse_paper_local(storage_path)
        if doi:
            crossref_meta = enhance_via_crossref(doi)
            if crossref_meta:
                if not parse_result.title and crossref_meta.title:
                    parse_result.title = crossref_meta.title
                if crossref_meta.abstract and not parse_result.abstract:
                    parse_result.abstract = crossref_meta.abstract
                if crossref_meta.journal:
                    parse_result.journal = crossref_meta.journal
        if pmid:
            pubmed_meta = enhance_via_pubmed(pmid)
            if pubmed_meta:
                if not parse_result.title and pubmed_meta.title:
                    parse_result.title = pubmed_meta.title

        # Step 2: Extract medical entities
        full_text = (parse_result.abstract or "") + " " + " ".join(
            s.content for s in parse_result.sections
        )
        entities = extract_medical_entities(full_text)

        # Step 3: Update Paper record with parsed data
        import asyncio
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

        asyncio.run(_update_paper())

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

        # Step 5: Embed chunks
        batch_size = 10
        all_vectors = []
        for i in range(0, len(paper_chunks), batch_size):
            batch = [c["content"] for c in paper_chunks[i:i + batch_size]]
            vectors = embed_texts(batch)
            all_vectors.extend(vectors)

        # Step 6: Write to Weaviate
        client = get_client()
        client.connect()
        try:
            collection = client.collections.get(COLLECTION_NAME)
            chunk_ids = []

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
                collection.data.insert(
                    uuid=weaviate_uuid,
                    properties=properties,
                    vector=vector,
                )
                chunk_ids.append(weaviate_uuid)
        finally:
            client.close()

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

        asyncio.run(_finalize())

        return {
            "paper_id": paper_id,
            "document_id": document_id,
            "kb_id": resolved_kb_id or "",
            "status": "ready",
            "chunk_count": len(paper_chunks),
            "chunk_ids": chunk_ids,
        }

    except Exception as e:
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
    import asyncio
    from sqlalchemy import select
    from app.models.audit import EvaluationQuestion, EvaluationRun
    from app.services.rag import hybrid_search

    try:
        async def _run():
            async with async_session() as session:
                # Update run status to running
                await session.execute(
                    update(EvaluationRun)
                    .where(EvaluationRun.id == run_id)
                    .values(status="running")
                )
                await session.commit()

                # Get questions
                q_stmt = select(EvaluationQuestion).where(
                    EvaluationQuestion.eval_set_id == eval_set_id,
                    EvaluationQuestion.org_id == org_id,
                )
                result = await session.execute(q_stmt)
                questions = list(result.scalars().all())

                metrics = {
                    "total_questions": len(questions),
                    "answered": 0,
                    "zero_result": 0,
                    "correct_kb_hit": 0,
                    "correct_doc_hit": 0,
                }

                for q in questions:
                    # Run search
                    sources = hybrid_search(
                        query=q.question,
                        org_id=org_id,
                        kb_ids=q.expected_kb_ids or [],
                        top_k=10,
                    )
                    retrieved_doc_ids = {s.document_id for s in sources}

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
                return metrics

        return asyncio.run(_run())

    except Exception as exc:
        error_message = str(exc)

        async def _fail():
            async with async_session() as session:
                await session.execute(
                    update(EvaluationRun)
                    .where(EvaluationRun.id == run_id)
                    .values(status="failed", error_message=error_message)
                )
                await session.commit()

        import asyncio
        asyncio.run(_fail())
        raise self.retry(exc=exc, countdown=60)
