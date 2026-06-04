import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import create_access_token, verify_password
from app.database import get_db
from app.models.tenant import User
from app.schemas.auth import LoginRequest, LoginResponse, UserResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
async def login(data: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate user and return JWT token."""
    logger.info("Login attempt username=%s", data.username)
    result = await db.execute(select(User).where(User.username == data.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(data.password, user.hashed_password):
        logger.warning("Login failed username=%s reason=invalid_credentials", data.username)
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    if not user.is_active:
        logger.warning("Login failed username=%s user_id=%s reason=inactive", data.username, user.id)
        raise HTTPException(status_code=403, detail="账号已被禁用")

    try:
        roles = json.loads(user.roles) if user.roles else []
    except (json.JSONDecodeError, TypeError):
        logger.warning("User roles JSON parse failed username=%s user_id=%s", data.username, user.id, exc_info=True)
        roles = []

    token = create_access_token(
        data={
            "sub": str(user.id),
            "org_id": str(user.org_id),
            "roles": roles,
        }
    )

    logger.info("Login succeeded username=%s user_id=%s org_id=%s roles=%s", user.username, user.id, user.org_id, roles)
    return LoginResponse(
        access_token=token,
        user=UserResponse(
            id=str(user.id),
            username=user.username,
            email=user.email,
            org_id=str(user.org_id),
            roles=roles,
        ),
    )
