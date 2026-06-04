"""Tests for audit log service."""
import pytest
import uuid
from unittest.mock import patch, AsyncMock, MagicMock

from app.services.audit import write_audit_log, query_audit_logs


class TestWriteAuditLog:
    @pytest.mark.asyncio
    async def test_write_audit_log(self):
        """Test writing an audit log entry."""
        with patch("app.services.audit.async_session") as mock_session_factory:
            mock_session = MagicMock()
            mock_session.commit = AsyncMock()
            mock_cm = MagicMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cm.__aexit__ = AsyncMock(return_value=False)
            mock_session_factory.return_value = mock_cm

            await write_audit_log(
                org_id="org-1",
                user_id="user-1",
                action="search",
                resource_type="kb",
                resource_id=str(uuid.uuid4()),
                details={"query": "test"},
                ip_address="127.0.0.1",
                status_code=200,
            )

            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_audit_log_minimal(self):
        """Test writing audit log with minimal fields."""
        with patch("app.services.audit.async_session") as mock_session_factory:
            mock_session = MagicMock()
            mock_session.commit = AsyncMock()
            mock_cm = MagicMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cm.__aexit__ = AsyncMock(return_value=False)
            mock_session_factory.return_value = mock_cm

            await write_audit_log(
                org_id="org-1",
                user_id="user-1",
                action="login",
            )

            mock_session.add.assert_called_once()
            entry = mock_session.add.call_args[0][0]
            assert entry.action == "login"
            assert entry.details == {}


class TestQueryAuditLogs:
    @pytest.mark.asyncio
    async def test_query_all(self):
        """Test querying all audit logs."""
        with patch("app.services.audit.async_session") as mock_session_factory:
            mock_session = MagicMock()
            mock_cm = MagicMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cm.__aexit__ = AsyncMock(return_value=False)
            mock_session_factory.return_value = mock_cm

            mock_result = MagicMock()
            mock_result.scalar.return_value = 5
            mock_result.scalars.return_value.all.return_value = []
            mock_session.execute = AsyncMock(return_value=mock_result)

            items, total = await query_audit_logs(org_id="org-1")

            assert total == 5
            assert items == []

    @pytest.mark.asyncio
    async def test_query_with_filters(self):
        """Test querying with action filter."""
        with patch("app.services.audit.async_session") as mock_session_factory:
            mock_session = MagicMock()
            mock_cm = MagicMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cm.__aexit__ = AsyncMock(return_value=False)
            mock_session_factory.return_value = mock_cm

            mock_result = MagicMock()
            mock_result.scalar.return_value = 2
            mock_result.scalars.return_value.all.return_value = []
            mock_session.execute = AsyncMock(return_value=mock_result)

            items, total = await query_audit_logs(
                org_id="org-1",
                action="search",
                resource_type="kb",
            )

            assert total == 2

    @pytest.mark.asyncio
    async def test_query_with_pagination(self):
        """Test querying with limit and offset."""
        with patch("app.services.audit.async_session") as mock_session_factory:
            mock_session = MagicMock()
            mock_cm = MagicMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cm.__aexit__ = AsyncMock(return_value=False)
            mock_session_factory.return_value = mock_cm

            mock_result = MagicMock()
            mock_result.scalar.return_value = 100
            mock_result.scalars.return_value.all.return_value = []
            mock_session.execute = AsyncMock(return_value=mock_result)

            items, total = await query_audit_logs(
                org_id="org-1",
                limit=10,
                offset=20,
            )

            assert total == 100
