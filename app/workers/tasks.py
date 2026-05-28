import uuid
from datetime import datetime, timezone

from sqlalchemy import update

from app.database import async_session
from app.models.chunk import DocumentChunk
from app.models.document import Document, DocumentVersion
from app.services.chunker import chunk_text
from app.services.embedding import embed_texts
from app.services.file_parser import parse_file
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
    from app.config import settings
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
                    "embedding_model": "text-embedding-v3",
                    "created_at": datetime.now(timezone.utc),
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
