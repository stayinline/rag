"""Tests for auth module."""
import uuid
from datetime import datetime, timezone, timedelta

from app.auth import (
    create_access_token,
    hash_password,
    verify_password,
    decode_access_token,
    get_current_user_from_token,
)


def test_hash_and_verify_password():
    plain = "pwd123"
    hashed = hash_password(plain)
    assert hashed != plain
    assert verify_password(plain, hashed)
    assert not verify_password("wrongpwd", hashed)


def test_create_access_token():
    data = {"sub": "user-1", "org_id": "org-1", "roles": ["admin"]}
    token = create_access_token(data, expires_delta=timedelta(minutes=30))
    assert isinstance(token, str)
    assert len(token) > 0


def test_decode_access_token():
    data = {"sub": "user-1", "org_id": "org-1", "roles": ["admin"]}
    token = create_access_token(data, expires_delta=timedelta(minutes=30))
    decoded = decode_access_token(token)
    assert decoded["sub"] == "user-1"
    assert decoded["org_id"] == "org-1"
    assert decoded["roles"] == ["admin"]


def test_token_expiration():
    data = {"sub": "user-1"}
    token = create_access_token(data, expires_delta=timedelta(seconds=-1))
    # Token is already expired, decoding should raise ExpiredSignatureError
    from jose.exceptions import ExpiredSignatureError
    try:
        decode_access_token(token)
        assert False, "Should have raised ExpiredSignatureError"
    except ExpiredSignatureError:
        pass  # Expected


def test_get_current_user_from_token():
    user_id = str(uuid.uuid4())
    org_id = str(uuid.uuid4())
    token = create_access_token({
        "sub": user_id,
        "org_id": org_id,
        "roles": ["viewer", "editor"],
    })
    user = get_current_user_from_token(token)
    assert user["user_id"] == user_id
    assert user["org_id"] == org_id
    assert user["roles"] == ["viewer", "editor"]


def test_get_current_user_from_token_missing_fields():
    token = create_access_token({"other": "data"})
    user = get_current_user_from_token(token)
    assert user["user_id"] is None
    assert user["org_id"] is None
    assert user["roles"] == []
