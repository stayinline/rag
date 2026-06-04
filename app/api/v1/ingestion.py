import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database import get_db
from app.models.task import IngestionJob
from app.schemas.document import IngestionJobResponse

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
