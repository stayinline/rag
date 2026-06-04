import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy import update as sql_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database import get_db
from app.models.document import Document, DocumentVersion
from app.models.task import IngestionJob
from app.schemas.document import IngestionJobResponse
from app.workers.tasks import queue_document_ingestion

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingestion-jobs", tags=["ingestion"])


@router.get("/{job_id}", response_model=IngestionJobResponse)
async def get_ingestion_job(
    job_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    logger.info("Get ingestion job request org_id=%s user_id=%s job_id=%s", user["org_id"], user["user_id"], job_id)
    stmt = select(IngestionJob).where(
        IngestionJob.id == job_id,
        IngestionJob.org_id == user["org_id"],
    )
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()
    if not job:
        logger.warning(
            "Get ingestion job failed org_id=%s user_id=%s job_id=%s reason=not_found",
            user["org_id"],
            user["user_id"],
            job_id,
        )
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    logger.info(
        "Get ingestion job succeeded org_id=%s user_id=%s job_id=%s status=%s type=%s",
        user["org_id"],
        user["user_id"],
        job_id,
        job.status,
        job.job_type,
    )
    return job


@router.post("/{job_id}/retry", response_model=IngestionJobResponse)
async def retry_ingestion_job(
    job_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    logger.info("Retry ingestion job request org_id=%s user_id=%s job_id=%s", user["org_id"], user["user_id"], job_id)
    stmt = select(IngestionJob).where(
        IngestionJob.id == job_id,
        IngestionJob.org_id == user["org_id"],
    )
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()
    if not job:
        logger.warning(
            "Retry ingestion job failed org_id=%s user_id=%s job_id=%s reason=not_found",
            user["org_id"],
            user["user_id"],
            job_id,
        )
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    if job.job_type != "parse":
        raise HTTPException(status_code=400, detail="Only parse ingestion jobs can be retried")
    if job.status == "running":
        raise HTTPException(status_code=409, detail="Ingestion job is already running")

    version = await db.get(DocumentVersion, job.version_id)
    document = await db.get(Document, job.document_id)
    if not version or not document:
        raise HTTPException(status_code=404, detail="Document or version not found")

    await db.execute(
        sql_update(IngestionJob)
        .where(IngestionJob.id == job.id)
        .values(status="pending", error_code=None, error_message=None, finished_at=None)
    )
    await db.execute(sql_update(Document).where(Document.id == document.id).values(status="draft"))
    await db.execute(sql_update(DocumentVersion).where(DocumentVersion.id == version.id).values(index_status="pending"))
    await db.commit()
    await db.refresh(job)

    try:
        async_result = queue_document_ingestion(
            org_id=str(job.org_id),
            document_id=str(document.id),
            version_id=str(version.id),
            kb_id=str(document.kb_id),
            title=document.title,
            storage_path=version.storage_path,
        )
        logger.info(
            "Retry ingestion job queued org_id=%s user_id=%s job_id=%s document_id=%s root_task_id=%s",
            user["org_id"],
            user["user_id"],
            job.id,
            document.id,
            async_result.id,
        )
    except Exception as exc:
        logger.exception("Retry ingestion job queue failed org_id=%s user_id=%s job_id=%s", user["org_id"], user["user_id"], job_id)
        await db.execute(
            sql_update(IngestionJob)
            .where(IngestionJob.id == job.id)
            .values(status="failed", error_code=exc.__class__.__name__[:100], error_message=str(exc)[:2000])
        )
        await db.execute(sql_update(Document).where(Document.id == document.id).values(status="failed"))
        await db.execute(sql_update(DocumentVersion).where(DocumentVersion.id == version.id).values(index_status="failed"))
        await db.commit()
        raise HTTPException(status_code=503, detail="Document ingestion queue failed") from exc

    await db.refresh(job)
    return job
