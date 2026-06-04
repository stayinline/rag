import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database import get_db
from app.models.kb import KnowledgeBase
from app.schemas.kb import KBCreate, KBListResponse, KBResponse, KBUpdate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/kbs", tags=["knowledge-bases"])


@router.post("", response_model=KBResponse, status_code=201)
async def create_kb(
    data: KBCreate,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    logger.info("Create KB request org_id=%s user_id=%s name=%s", user["org_id"], user["user_id"], data.name)
    kb = KnowledgeBase(
        org_id=user["org_id"],
        name=data.name,
        description=data.description,
        created_by=user["user_id"],
    )
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    logger.info("Create KB succeeded org_id=%s user_id=%s kb_id=%s", user["org_id"], user["user_id"], kb.id)
    return kb


@router.get("", response_model=KBListResponse)
async def list_kbs(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    logger.info("List KBs request org_id=%s user_id=%s", user["org_id"], user["user_id"])
    stmt = select(KnowledgeBase).where(
        KnowledgeBase.org_id == user["org_id"],
        KnowledgeBase.deleted_at.is_(None),
    )
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar() or 0

    result = await db.execute(stmt.order_by(KnowledgeBase.created_at.desc()))
    items = result.scalars().all()
    logger.info("List KBs complete org_id=%s user_id=%s total=%s returned=%s", user["org_id"], user["user_id"], total, len(items))
    return {"items": items, "total": total}


@router.get("/{kb_id}", response_model=KBResponse)
async def get_kb(
    kb_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    logger.info("Get KB request org_id=%s user_id=%s kb_id=%s", user["org_id"], user["user_id"], kb_id)
    stmt = select(KnowledgeBase).where(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.org_id == user["org_id"],
    )
    result = await db.execute(stmt)
    kb = result.scalar_one_or_none()
    if not kb:
        logger.warning("Get KB failed org_id=%s user_id=%s kb_id=%s reason=not_found", user["org_id"], user["user_id"], kb_id)
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    logger.info("Get KB succeeded org_id=%s user_id=%s kb_id=%s", user["org_id"], user["user_id"], kb_id)
    return kb


@router.patch("/{kb_id}", response_model=KBResponse)
async def update_kb(
    kb_id: str,
    data: KBUpdate,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    logger.info(
        "Update KB request org_id=%s user_id=%s kb_id=%s fields=%s",
        user["org_id"],
        user["user_id"],
        kb_id,
        list(data.model_dump(exclude_unset=True).keys()),
    )
    stmt = select(KnowledgeBase).where(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.org_id == user["org_id"],
    )
    result = await db.execute(stmt)
    kb = result.scalar_one_or_none()
    if not kb:
        logger.warning("Update KB failed org_id=%s user_id=%s kb_id=%s reason=not_found", user["org_id"], user["user_id"], kb_id)
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    if data.name is not None:
        kb.name = data.name
    if data.description is not None:
        kb.description = data.description
    if data.is_active is not None:
        kb.is_active = data.is_active

    await db.commit()
    await db.refresh(kb)
    logger.info("Update KB succeeded org_id=%s user_id=%s kb_id=%s", user["org_id"], user["user_id"], kb_id)
    return kb


@router.delete("/{kb_id}", status_code=204)
async def delete_kb(
    kb_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    logger.info("Delete KB request org_id=%s user_id=%s kb_id=%s", user["org_id"], user["user_id"], kb_id)
    from sqlalchemy import update as sql_update

    stmt = select(KnowledgeBase).where(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.org_id == user["org_id"],
    )
    result = await db.execute(stmt)
    kb = result.scalar_one_or_none()
    if not kb:
        logger.warning("Delete KB failed org_id=%s user_id=%s kb_id=%s reason=not_found", user["org_id"], user["user_id"], kb_id)
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    await db.execute(
        sql_update(KnowledgeBase)
        .where(KnowledgeBase.id == kb_id)
        .values(is_active=False)
    )
    await db.commit()
    logger.info("Delete KB succeeded org_id=%s user_id=%s kb_id=%s", user["org_id"], user["user_id"], kb_id)
