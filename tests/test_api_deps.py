"""Tests for API dependencies (auth middleware, deps)."""
import uuid
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.api.deps import get_current_user, get_current_user_optional


@pytest.mark.asyncio
async def test_get_current_user_valid_token():
    user_id = str(uuid.uuid4())
    org_id = str(uuid.uuid4())

    with patch("app.api.deps.get_current_user_from_token") as mock:
        mock.return_value = {"user_id": user_id, "org_id": org_id, "roles": ["admin"]}
        user = await get_current_user("Bearer test-token")
        assert user["user_id"] == user_id
        assert user["org_id"] == org_id


@pytest.mark.asyncio
async def test_get_current_user_invalid_header():
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user("Invalid test-token")
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_missing_sub():
    with patch("app.api.deps.get_current_user_from_token") as mock:
        mock.return_value = {"org_id": "org-1", "roles": []}  # No user_id
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user("Bearer test-token")
        assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_decode_error():
    with patch("app.api.deps.get_current_user_from_token") as mock:
        mock.side_effect = Exception("Invalid token")
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user("Bearer test-token")
        assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_optional_no_auth():
    user = await get_current_user_optional(None)
    assert user is None


@pytest.mark.asyncio
async def test_get_current_user_optional_valid():
    user_id = str(uuid.uuid4())
    with patch("app.api.deps.get_current_user_from_token") as mock:
        mock.return_value = {"user_id": user_id, "org_id": "org-1"}
        user = await get_current_user_optional("Bearer test-token")
        assert user is not None
        assert user["user_id"] == user_id


@pytest.mark.asyncio
async def test_get_current_user_optional_invalid():
    with patch("app.api.deps.get_current_user_from_token") as mock:
        mock.side_effect = Exception("Bad token")
        user = await get_current_user_optional("Bearer bad-token")
        assert user is None
