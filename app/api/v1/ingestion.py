from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database import get_db
from app.models.task import IngestionJob
from app.schemas.document import IngestionJobResponse

router = APIRouter(prefix="/ingestion-jobs", tags=["ingestion"])


@router.get("/{job_id}", response_model=IngestionJobResponse)
async def get_ingestion_job(
    job_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(IngestionJob).where(
        IngestionJob.id == job_id,
        IngestionJob.org_id == user["org_id"],
    )
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return job
