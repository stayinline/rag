from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database import get_db
from app.models.kb import KnowledgeBase
from app.schemas.kb import KBCreate, KBListResponse, KBResponse, KBUpdate

router = APIRouter(prefix="/kbs", tags=["knowledge-bases"])


@router.post("", response_model=KBResponse, status_code=201)
async def create_kb(
    data: KBCreate,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    kb = KnowledgeBase(
        org_id=user["org_id"],
        name=data.name,
        description=data.description,
        created_by=user["user_id"],
    )
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    return kb


@router.get("", response_model=KBListResponse)
async def list_kbs(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(KnowledgeBase).where(
        KnowledgeBase.org_id == user["org_id"],
        KnowledgeBase.deleted_at.is_(None),
    )
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar() or 0

    result = await db.execute(stmt.order_by(KnowledgeBase.created_at.desc()))
    items = result.scalars().all()
    return {"items": items, "total": total}


@router.get("/{kb_id}", response_model=KBResponse)
async def get_kb(
    kb_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(KnowledgeBase).where(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.org_id == user["org_id"],
    )
    result = await db.execute(stmt)
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


@router.patch("/{kb_id}", response_model=KBResponse)
async def update_kb(
    kb_id: str,
    data: KBUpdate,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(KnowledgeBase).where(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.org_id == user["org_id"],
    )
    result = await db.execute(stmt)
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    if data.name is not None:
        kb.name = data.name
    if data.description is not None:
        kb.description = data.description
    if data.is_active is not None:
        kb.is_active = data.is_active

    await db.commit()
    await db.refresh(kb)
    return kb


@router.delete("/{kb_id}", status_code=204)
async def delete_kb(
    kb_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import update as sql_update

    stmt = select(KnowledgeBase).where(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.org_id == user["org_id"],
    )
    result = await db.execute(stmt)
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    await db.execute(
        sql_update(KnowledgeBase)
        .where(KnowledgeBase.id == kb_id)
        .values(is_active=False)
    )
    await db.commit()
