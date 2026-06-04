import logging
import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import update as sql_update
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.config import settings
from app.database import get_db
from app.models.document import Document, DocumentVersion
from app.models.task import IngestionJob
from app.schemas.document import DocumentListResponse, DocumentResponse
from app.workers.tasks import queue_document_ingestion

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["documents"])


def make_document_ingestion_key(
    org_id: str,
    document_id: str,
    version_id: str,
    job_type: str,
    content_hash: str,
) -> str:
    raw_key = f"{org_id}:{document_id}:{version_id}:{job_type}:{content_hash}"
    return f"{job_type}:{uuid.uuid5(uuid.NAMESPACE_URL, raw_key)}"


@router.post("/kbs/{kb_id}/documents", response_model=DocumentResponse, status_code=201)
async def upload_document(
    kb_id: str,
    file: UploadFile,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    logger.info(
        "Upload document request org_id=%s user_id=%s kb_id=%s filename=%s content_type=%s",
        user["org_id"],
        user["user_id"],
        kb_id,
        file.filename,
        file.content_type,
    )
    # Verify KB exists
    from app.models.kb import KnowledgeBase

    kb_stmt = select(KnowledgeBase).where(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.org_id == user["org_id"],
    )
    kb_result = await db.execute(kb_stmt)
    if not kb_result.scalar_one_or_none():
        logger.warning(
            "Upload document failed org_id=%s user_id=%s kb_id=%s reason=kb_not_found",
            user["org_id"],
            user["user_id"],
            kb_id,
        )
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Save file
    file_ext = os.path.splitext(file.filename or "")[1].lower()
    file_id = str(uuid.uuid4())
    filename = f"{file_id}{file_ext}"
    file_dir = os.path.join(settings.storage_path, str(user["org_id"]), kb_id)
    os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, filename)

    content = await file.read()
    logger.info(
        "Upload document file read org_id=%s user_id=%s kb_id=%s filename=%s size_bytes=%s extension=%s",
        user["org_id"],
        user["user_id"],
        kb_id,
        file.filename,
        len(content),
        file_ext,
    )
    with open(file_path, "wb") as f:
        f.write(content)
    logger.debug("Upload document file saved path=%s", file_path)

    # Create document record
    import hashlib

    content_hash = hashlib.sha256(content).hexdigest()
    document = Document(
        org_id=user["org_id"],
        kb_id=kb_id,
        title=file.filename or filename,
        file_name=file.filename,
        file_type=file_ext.lstrip("."),
        content_hash=content_hash,
        document_type="general",
        created_by=user["user_id"],
    )
    db.add(document)
    await db.flush()

    # Create version record
    version = DocumentVersion(
        org_id=user["org_id"],
        document_id=document.id,
        version=1,
        storage_path=file_path,
        content_hash=content_hash,
    )
    db.add(version)
    await db.flush()

    # Create ingestion job
    job = IngestionJob(
        org_id=user["org_id"],
        document_id=document.id,
        version_id=version.id,
        job_type="parse",
        idempotency_key=make_document_ingestion_key(
            org_id=str(user["org_id"]),
            document_id=str(document.id),
            version_id=str(version.id),
            job_type="parse",
            content_hash=content_hash,
        ),
    )
    db.add(job)
    await db.commit()
    await db.refresh(document)
    logger.info(
        "Upload document records created org_id=%s user_id=%s kb_id=%s document_id=%s version_id=%s job_id=%s content_hash=%s",
        user["org_id"],
        user["user_id"],
        kb_id,
        document.id,
        version.id,
        job.id,
        content_hash,
    )

    try:
        async_result = queue_document_ingestion(
            org_id=str(user["org_id"]),
            document_id=str(document.id),
            version_id=str(version.id),
            kb_id=str(kb_id),
            title=document.title,
            storage_path=file_path,
        )
        logger.info(
            "Upload document ingestion chain queued org_id=%s user_id=%s kb_id=%s document_id=%s version_id=%s job_id=%s root_task_id=%s",
            user["org_id"],
            user["user_id"],
            kb_id,
            document.id,
            version.id,
            job.id,
            async_result.id,
        )
    except Exception as exc:
        logger.exception(
            "Upload document ingestion queue failed org_id=%s user_id=%s kb_id=%s document_id=%s version_id=%s job_id=%s",
            user["org_id"],
            user["user_id"],
            kb_id,
            document.id,
            version.id,
            job.id,
        )
        await db.execute(sql_update(Document).where(Document.id == document.id).values(status="failed"))
        await db.execute(sql_update(DocumentVersion).where(DocumentVersion.id == version.id).values(index_status="failed"))
        await db.execute(
            sql_update(IngestionJob)
            .where(IngestionJob.id == job.id)
            .values(
                status="failed",
                error_code=exc.__class__.__name__[:100],
                error_message=str(exc)[:2000],
                finished_at=datetime.now(timezone.utc),
            )
        )
        await db.commit()
        raise HTTPException(status_code=503, detail="Document ingestion queue failed") from exc

    return document


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    logger.info("Get document request org_id=%s user_id=%s document_id=%s", user["org_id"], user["user_id"], document_id)
    stmt = select(Document).where(
        Document.id == document_id,
        Document.org_id == user["org_id"],
    )
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        logger.warning(
            "Get document failed org_id=%s user_id=%s document_id=%s reason=not_found",
            user["org_id"],
            user["user_id"],
            document_id,
        )
        raise HTTPException(status_code=404, detail="Document not found")
    logger.info(
        "Get document succeeded org_id=%s user_id=%s document_id=%s status=%s",
        user["org_id"],
        user["user_id"],
        document_id,
        doc.status,
    )
    return doc


@router.get("/kbs/{kb_id}/documents", response_model=DocumentListResponse)
async def list_documents(
    kb_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    logger.info("List documents request org_id=%s user_id=%s kb_id=%s", user["org_id"], user["user_id"], kb_id)
    stmt = select(Document).where(
        Document.kb_id == kb_id,
        Document.org_id == user["org_id"],
        Document.deleted_at.is_(None),
    )
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar() or 0

    result = await db.execute(stmt.order_by(Document.created_at.desc()).limit(50))
    items = result.scalars().all()
    logger.info(
        "List documents complete org_id=%s user_id=%s kb_id=%s total=%s returned=%s",
        user["org_id"],
        user["user_id"],
        kb_id,
        total,
        len(items),
    )
    return {"items": items, "total": total}


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    logger.info("Delete document request org_id=%s user_id=%s document_id=%s", user["org_id"], user["user_id"], document_id)
    from datetime import datetime, timezone
    from sqlalchemy import update as sql_update

    stmt = select(Document).where(
        Document.id == document_id,
        Document.org_id == user["org_id"],
    )
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        logger.warning(
            "Delete document failed org_id=%s user_id=%s document_id=%s reason=not_found",
            user["org_id"],
            user["user_id"],
            document_id,
        )
        raise HTTPException(status_code=404, detail="Document not found")

    await db.execute(
        sql_update(Document)
        .where(Document.id == document_id)
        .values(deleted_at=datetime.now(timezone.utc), status="deleted")
    )
    await db.commit()
    logger.info("Delete document succeeded org_id=%s user_id=%s document_id=%s", user["org_id"], user["user_id"], document_id)
