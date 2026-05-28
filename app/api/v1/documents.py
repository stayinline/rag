import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.config import settings
from app.database import get_db
from app.models.document import Document, DocumentVersion
from app.models.task import IngestionJob
from app.schemas.document import DocumentListResponse, DocumentResponse
from app.workers.tasks import parse_document_task

router = APIRouter(prefix="", tags=["documents"])


@router.post("/kbs/{kb_id}/documents", response_model=DocumentResponse, status_code=201)
async def upload_document(
    kb_id: str,
    file: UploadFile,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Verify KB exists
    from app.models.kb import KnowledgeBase

    kb_stmt = select(KnowledgeBase).where(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.org_id == user["org_id"],
    )
    kb_result = await db.execute(kb_stmt)
    if not kb_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Save file
    file_ext = os.path.splitext(file.filename or "")[1].lower()
    file_id = str(uuid.uuid4())
    filename = f"{file_id}{file_ext}"
    file_dir = os.path.join(settings.storage_path, str(user["org_id"]), kb_id)
    os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

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
        idempotency_key=f"{user['org_id']}:{document.id}:{version.id}:parse:{content_hash}",
    )
    db.add(job)
    await db.commit()
    await db.refresh(document)

    # Kick off Celery parse task
    parse_document_task.delay(
        org_id=str(user["org_id"]),
        document_id=str(document.id),
        version_id=str(version.id),
        storage_path=file_path,
    )

    return document


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Document).where(
        Document.id == document_id,
        Document.org_id == user["org_id"],
    )
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.get("/kbs/{kb_id}/documents", response_model=DocumentListResponse)
async def list_documents(
    kb_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Document).where(
        Document.kb_id == kb_id,
        Document.org_id == user["org_id"],
        Document.deleted_at.is_(None),
    )
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar() or 0

    result = await db.execute(stmt.order_by(Document.created_at.desc()).limit(50))
    items = result.scalars().all()
    return {"items": items, "total": total}


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from datetime import datetime, timezone
    from sqlalchemy import update as sql_update

    stmt = select(Document).where(
        Document.id == document_id,
        Document.org_id == user["org_id"],
    )
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    await db.execute(
        sql_update(Document)
        .where(Document.id == document_id)
        .values(deleted_at=datetime.now(timezone.utc), status="deleted")
    )
    await db.commit()
