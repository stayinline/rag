"""Tests for Weaviate client."""
from unittest.mock import MagicMock, patch

from app.services.weaviate_client import (
    COLLECTION_NAME,
    COLLECTION_PROPERTIES,
    ensure_collection,
    get_client,
)


def test_collection_name():
    assert COLLECTION_NAME == "KnowledgeChunk"


def test_collection_properties():
    assert len(COLLECTION_PROPERTIES) > 0
    prop_names = [p.name for p in COLLECTION_PROPERTIES]
    assert "org_id" in prop_names
    assert "kb_id" in prop_names
    assert "document_id" in prop_names
    assert "content" in prop_names
    assert "status" in prop_names
    assert "security_level" in prop_names


def test_ensure_collection_creates():
    client = MagicMock()
    client.collections.exists = MagicMock(return_value=False)
    client.collections.create = MagicMock()

    ensure_collection(client)
    client.collections.create.assert_called_once()
    call_kwargs = client.collections.create.call_args[1]
    assert call_kwargs["name"] == "KnowledgeChunk"


def test_ensure_collection_skips_existing():
    client = MagicMock()
    client.collections.exists = MagicMock(return_value=True)
    client.collections.create = MagicMock()

    ensure_collection(client)
    client.collections.create.assert_not_called()


def test_get_client(mock_settings):
    with patch("app.services.weaviate_client.weaviate.connect_to_local") as mock:
        mock.return_value = MagicMock()
        client = get_client()
        mock.assert_called_once()
