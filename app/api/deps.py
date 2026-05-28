from fastapi import Header, HTTPException

from app.auth import get_current_user_from_token


async def get_current_user(authorization: str = Header(...)) -> dict:
    """Extract and validate JWT from Authorization header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization[7:]
    try:
        user = get_current_user_from_token(token)
        if not user.get("user_id"):
            raise HTTPException(status_code=401, detail="Invalid token")
        return user
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")


async def get_current_user_optional(authorization: str | None = Header(None)) -> dict | None:
    """Optional authentication for public endpoints."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        return get_current_user_from_token(authorization[7:])
    except Exception:
        return None
