import logging

from fastapi import Header, HTTPException

from app.auth import get_current_user_from_token

logger = logging.getLogger(__name__)


async def get_current_user(authorization: str = Header(...)) -> dict:
    """Extract and validate JWT from Authorization header."""
    if not authorization.startswith("Bearer "):
        logger.warning("Authentication rejected: invalid authorization header format")
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization[7:]
    try:
        user = get_current_user_from_token(token)
        if not user.get("user_id"):
            logger.warning("Authentication rejected: token has no user_id org_id=%s", user.get("org_id"))
            raise HTTPException(status_code=401, detail="Invalid token")
        logger.debug(
            "Authentication accepted user_id=%s org_id=%s roles=%s",
            user.get("user_id"),
            user.get("org_id"),
            user.get("roles", []),
        )
        return user
    except HTTPException:
        raise
    except Exception:
        logger.warning("Authentication rejected: token validation failed", exc_info=True)
        raise HTTPException(status_code=401, detail="Authentication failed")


async def get_current_user_optional(authorization: str | None = Header(None)) -> dict | None:
    """Optional authentication for public endpoints."""
    if not authorization or not authorization.startswith("Bearer "):
        logger.debug("Optional authentication skipped: no bearer token")
        return None
    try:
        user = get_current_user_from_token(authorization[7:])
        logger.debug("Optional authentication accepted user_id=%s org_id=%s", user.get("user_id"), user.get("org_id"))
        return user
    except Exception:
        logger.warning("Optional authentication token validation failed", exc_info=True)
        return None
